"""Qwen3.8-Flash-Next decoder stack (text-only).

The residual state is ``R [T, hc_count*hidden]`` end to end: the embedding is repeated over the
``hc_count`` streams, every layer mixes them down to one ``[T, hidden]`` block input and injects
its output back, and the top-level mixer collapses them once before ``lm_head``. There is no
input/post layernorm and no final ``model.norm`` -- the hyper-connection norms are the only ones.

Layer contract (frozen): ``forward(R [T, hc*hidden], batch) -> R' [T, hc*hidden]`` with an
immediate combine::

    R  = R + ple(R, batch)                 # zero-based layer 1 only
    x, s = attn_hc.mix(R); y = (GDN | QSA)(x); R = attn_hc.combine(R, y, s)
    x, s = mlp_hc.mix(R);  y = MoE(x);        R = mlp_hc.combine(R, y, s)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List
import os
import time

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    LinearReplicated,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.layers.quantization import LayerKind, QuantConfig
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Qwen4ExpAttention
from .hc import GatedResidual, GroupedPlusOneRMSNorm
from .moe import Qwen4ExpMoE
from .ple import PLELayer
from freetoken.models.blocks import embed_input_ids
from freetoken.models.qwen3_vl.vision import Qwen3VLVisionModel, QwenVLVisionMixin

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


_LAYER_TIMING_ENV = "FREETOKEN_DEBUG_LAYER_TIMING"
_layer_timing_enabled = os.getenv(_LAYER_TIMING_ENV, "0").strip().lower() not in {"0", "false", ""}
# accumulated ms per component; not touched unless FREETOKEN_DEBUG_LAYER_TIMING is set, so it
# costs nothing on a normal run
_layer_timing_totals: dict[str, float] = {"mixer": 0.0, "moe": 0.0, "calls": 0.0}


def build_linear_mixer(config: ModelConfig, layer_id: int, prefix: str) -> BaseOP:
    """GDN mixer of a linear_attention layer (Qwen3.5's GDN with a configurable output gate)."""
    from .gdn import Qwen4ExpGatedDeltaNet

    g = config.linear_attention_group()
    return Qwen4ExpGatedDeltaNet(
        hidden_size=config.hidden_size,
        num_k_heads=g.num_key_heads,
        num_v_heads=g.num_value_heads,
        head_k_dim=g.key_head_dim,
        head_v_dim=g.value_head_dim,
        conv_kernel_size=g.conv_kernel_dim,
        rms_norm_eps=config.rms_norm_eps,
        layer_id=layer_id,
        output_gate=g.output_gate,
        quant_config=config.quant,
        prefix=prefix,
    )


class Qwen4ExpDecoderLayer(BaseOP):
    """One decoder layer over the hyper-connection streams (see the module docstring for the flow)."""

    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        prefix: str = "",
        moe_layer_id: int | None = None,
    ) -> None:
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            self.linear_attn = build_linear_mixer(config, layer_id, f"{prefix}.linear_attn")
        else:
            self.self_attn = Qwen4ExpAttention(config, layer_id, prefix=f"{prefix}.self_attn")
        # The MTP draft layer's MoE block indexes the offload cache by bank, not by its own
        # KV/attention layer_id (one past the target stack): its own bank when registered
        # (ModelConfig.mtp_expert_bank), else a target bank -- moe_layer_id carries that.
        self.mlp = Qwen4ExpMoE(
            config, moe_layer_id if moe_layer_id is not None else layer_id, prefix=f"{prefix}.mlp"
        )
        self.attn_hyper_connection = GatedResidual(config, prefix=f"{prefix}.attn_hyper_connection")
        self.mlp_hyper_connection = GatedResidual(config, prefix=f"{prefix}.mlp_hyper_connection")
        self.ple = (
            PLELayer(config, layer_id, prefix=f"{prefix}.ple")
            if layer_id in config.qwen4_args.ple_layer_ids
            else None
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, hidden: torch.Tensor, batch: Batch, *, last_only: bool = False
    ) -> torch.Tensor:
        if self.ple is not None:
            hidden = hidden + self.ple.forward(hidden, batch)
        block_input, inject = self.attn_hyper_connection.mix(hidden)
        if not _layer_timing_enabled:
            if self._is_linear:
                block_output = self.linear_attn.forward(block_input)
            else:
                block_output = self.self_attn.forward(block_input, batch)
            hidden = self.attn_hyper_connection.combine(hidden, block_output, inject)
            if last_only:
                hidden = hidden[-1:]
            block_input, inject = self.mlp_hyper_connection.mix(hidden)
            return self.mlp_hyper_connection.combine(hidden, self.mlp.forward(block_input), inject)

        mixer_start, mixer_end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        moe_start, moe_end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        mixer_start.record()
        if self._is_linear:
            block_output = self.linear_attn.forward(block_input)
        else:
            block_output = self.self_attn.forward(block_input, batch)
        mixer_end.record()
        hidden = self.attn_hyper_connection.combine(hidden, block_output, inject)
        if last_only:
            hidden = hidden[-1:]
        block_input, inject = self.mlp_hyper_connection.mix(hidden)
        moe_start.record()
        moe_out = self.mlp.forward(block_input)
        moe_end.record()
        torch.cuda.current_stream().synchronize()
        _layer_timing_totals["mixer"] += mixer_start.elapsed_time(mixer_end)
        _layer_timing_totals["moe"] += moe_start.elapsed_time(moe_end)
        _layer_timing_totals["calls"] += 1
        if int(_layer_timing_totals["calls"]) % 256 == 0:
            print(f"[layer-timing] totals_ms={_layer_timing_totals}", flush=True)
        return self.mlp_hyper_connection.combine(hidden, moe_out, inject)

    def forward_last(self, hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
        return self.forward(hidden, batch, last_only=True)


class _MTPQuantConfig(QuantConfig):
    """Match the target's routed-bank layout while retaining draft dense-weight rules."""

    dialect = "qwen4-mtp"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config.quant.name_map, ())
        self._target = config.quant
        self._expert_prefix = f"model.layers.{config.first_k_dense_replace}.mlp.experts"

    @classmethod
    def claims(cls, q: dict) -> bool:
        return False

    def scheme_for_name(self, name: str):
        return self._target.scheme_for_name(name)

    def scheme_for(self, prefix: str):
        return self._target.scheme_for(prefix)

    def get_quant_method(self, layer, prefix: str):
        if layer.quant_layer_kind is LayerKind.MOE:
            prefix = self._expert_prefix
        return self._target.get_quant_method(layer, prefix)


class Qwen4ExpMTP(BaseOP):
    """One native draft layer over the target's final, unmixed residual streams.

    ``forward(R, next_ids, batch)`` returns the next wide residual; ``to_head`` reduces
    it for the shared LM head. KV metadata and the extra expert bank belong to the caller.
    This single-rank seam shares the target embedding and does not schedule draft steps.
    """

    def __init__(self, config: ModelConfig, layer_id: int, *, embedding: BaseOP) -> None:
        from dataclasses import replace

        from freetoken.models.config import FullAttentionGroupConfig

        mtp = config.qwen4_args.mtp
        if not mtp.enabled or mtp.num_hidden_layers != 1 or not mtp.hybrid:
            raise ValueError("Qwen4 MTP requires one native hybrid draft layer")
        if mtp.layer_types != ("full_attention",) or mtp.use_hidden_state_from_layer is not None:
            raise ValueError("Qwen4 MTP requires full attention and the final target residual")
        if mtp.rope_theta not in (None, config.rotary_config.base):
            raise ValueError("Qwen4 MTP with a separate RoPE base is not supported")
        if config.mtp_layer_id != layer_id or layer_id != config.num_layers:
            raise ValueError("register the MTP layer with with_mtp_layer before construction")
        if not isinstance(config.attention_group_for_layer(layer_id), FullAttentionGroupConfig):
            raise ValueError("Qwen4 MTP requires its own full-attention slot")
        self.hc_count = config.qwen4_args.hc_count
        self.hidden_size = hidden = config.hidden_size
        self.layer_id = layer_id
        self._embed_ref = embedding
        self._image_token_id = config.image_token_id
        self.pre_fc_norm_hidden = GroupedPlusOneRMSNorm(
            self.hc_count * hidden, config.rms_norm_eps, self.hc_count
        )
        self.pre_fc_norm_embedding = GroupedPlusOneRMSNorm(hidden, config.rms_norm_eps, 1)
        self.fc_hidden = LinearReplicated(
            hidden, hidden, has_bias=False, quant_config=config.quant, prefix="mtp.fc_hidden"
        )
        self.fc_embedding = LinearReplicated(
            hidden, hidden, has_bias=False, quant_config=config.quant, prefix="mtp.fc_embedding"
        )
        head_config = (
            replace(config, quant=_MTPQuantConfig(config)) if config.quant is not None else config
        )
        self.layers = OPList(
            [
                Qwen4ExpDecoderLayer(
                    head_config,
                    layer_id,
                    prefix="mtp.layers.0",
                    # own bank (the last MoE bank) when registered, else the target's first
                    moe_layer_id=(
                        config.num_moe_layers - 1
                        if config.mtp_expert_bank
                        else config.first_k_dense_replace
                    ),
                )
            ]
        )
        self.hyper_connection_mixer = GatedResidual(
            config, use_combine=False, prefix="mtp.hyper_connection_mixer"
        )

    def forward(self, residual: torch.Tensor, next_ids: torch.Tensor, batch: Batch) -> torch.Tensor:
        hidden = self._prepare_hidden(residual, next_ids)
        return self.layers.op_list[0].forward(hidden, batch)

    def forward_last(
        self, residual: torch.Tensor, next_ids: torch.Tensor, batch: Batch
    ) -> torch.Tensor:
        """Run row-wise attention/state work for all inputs and the MLP for the last."""
        hidden = self._prepare_hidden(residual, next_ids)
        return self.layers.op_list[0].forward_last(hidden, batch)

    def prime_kv(self, residual: torch.Tensor, next_ids: torch.Tensor, batch: Batch) -> None:
        """Build exact target-fed head KV without unused attention outputs or expert work."""
        if not hasattr(get_global_ctx().attn_backend, "store_qsa_kv"):
            self.forward(residual, next_ids, batch)
            return
        hidden = self._prepare_hidden(residual, next_ids)
        layer = self.layers.op_list[0]
        if layer.ple is not None:
            hidden = hidden + layer.ple.forward(hidden, batch)
        block_input, _ = layer.attn_hyper_connection.mix(hidden)
        layer.self_attn.prime_kv(block_input, batch)

    def _prepare_hidden(self, residual: torch.Tensor, next_ids: torch.Tensor) -> torch.Tensor:
        from freetoken.mm import restore_placeholder

        tokens = residual.shape[0]
        rn = self.pre_fc_norm_hidden.forward(residual)
        fh = self.fc_hidden.forward(rn.reshape(tokens * self.hc_count, self.hidden_size))
        fh = fh.reshape(tokens, self.hc_count * self.hidden_size)
        if self._image_token_id is not None:
            next_ids = restore_placeholder(next_ids, self._image_token_id)
        embedded = self._embed_ref.forward(next_ids).to(residual.dtype)
        fe = self.fc_embedding.forward(self.pre_fc_norm_embedding.forward(embedded))
        return fh + fe.repeat(1, self.hc_count)

    def mix(self, residual: torch.Tensor) -> torch.Tensor:
        return self.hyper_connection_mixer.mix(residual)[0]

    def to_head(self, residual: torch.Tensor) -> torch.Tensor:
        return self.mix(residual)


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model") -> None:
        self.hc_count = config.qwen4_args.hc_count
        self._capture_mtp_residual = config.mtp_layer_id is not None
        self._last_residual = None
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [
                Qwen4ExpDecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in range(config.num_layers)
            ]
        )
        self.hyper_connection_mixer = GatedResidual(
            config, use_combine=False, prefix=f"{prefix}.hyper_connection_mixer"
        )
        # plain tuple (not an OP child), so it never shows up in the state dict
        self._ple = tuple(layer.ple for layer in self.layers.op_list if layer.ple is not None)

    @property
    def ple_layers(self) -> List[PLELayer]:
        """The PLE layers in decoder order -- the seam the loader attaches table backends to."""
        return list(self._ple)

    def forward(self, input_ids: torch.Tensor, batch: Batch) -> torch.Tensor:
        hidden = embed_input_ids(self.embed_tokens, input_ids, batch)
        hidden = hidden.repeat(1, self.hc_count)
        meta = None
        if self._ple:
            from .ple import build_ple_metadata, commit_ngram_context

            meta = build_ple_metadata(batch, self._ple[0].args, input_ids.device)
            for ple in self._ple:  # gather the pinned-host PLE rows while the early layers run
                ple.start_prefetch(batch, meta)
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden, batch)
        if meta is not None:
            # single writer: the layers only read the context, so a second PLE layer's
            # prefetch sees the un-rolled window
            spec_out = None
            if getattr(batch, "spec_logits_indices", None) is not None:
                buffers = getattr(get_global_ctx().linear_state_pool, "spec_slot_states", {})
                if (
                    "ple_ngram_ctx" in buffers
                    and input_ids.shape[0] <= buffers["ple_ngram_ctx"].shape[1]
                ):
                    spec_out = buffers["ple_ngram_ctx"][0]
            commit_ngram_context(meta, getattr(batch, "fla_metadata", None), spec_out=spec_out)
        if self._capture_mtp_residual:
            self._last_residual = hidden
            commit_mtp_residual(hidden, batch)
        return self.hyper_connection_mixer.mix(hidden)[0]


def commit_mtp_residual(hidden: torch.Tensor, batch: Batch) -> None:
    """Cache the target residual at live and donated prefix boundaries for draft priming."""
    pool = get_global_ctx().linear_state_pool
    if pool is None or not pool.has_slot_state("mtp_residual"):
        return
    metadata = batch.fla_metadata
    state = pool.slot_state("mtp_residual")
    state.index_copy_(0, metadata.cache_indices.long(), hidden[metadata.cu_seqlens[1:].long() - 1])
    if metadata.track_dst is not None:
        state.index_copy_(0, metadata.track_dst, hidden[metadata.track_boundary_row - 1])
    if batch.spec_logits_indices is not None:
        rows = pool.spec_slot_states.get("mtp_residual")
        if rows is not None and hidden.shape[0] <= rows.shape[1]:
            rows[0, : hidden.shape[0]].copy_(hidden)


class Qwen4ExpForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.model = Qwen4ExpModel(config)
        self.mtp = (
            Qwen4ExpMTP(config, config.mtp_layer_id, embedding=self.model.embed_tokens)
            if config.mtp_layer_id is not None
            else None
        )
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        from .gguf import convert_qwen4exp_to_gguf, is_gguf_model

        if is_gguf_model(config):
            assert config.gguf_model_path is not None, (
                "expert_quant=='gguf' but ModelConfig.gguf_model_path is unset; the "
                "GGUF loader should have populated it."
            )
            convert_qwen4exp_to_gguf(self, config, model_path=config.gguf_model_path)
        super().__init__()

    def load_host_tables(self, engine_config) -> int:
        """Attach the PLE n-gram table (pinned checkpoint bank, or zeros for dummy weights); returns the pinned host bytes the engine reserves from its pin budget."""
        ple_layers = self.model.ple_layers
        if not ple_layers:
            return 0
        from .ple import PinnedUVATable, ZeroTable, derive_ngram_hash_constants

        if getattr(engine_config, "use_dummy_weight", False):
            # Dummy fill leaves the int64 hash buffers garbage (a zero vocab size divides by
            # zero in the hash), so re-derive the real constants and read a zero table.
            for ple in ple_layers:
                args = ple.args
                mult, sizes, offsets = derive_ngram_hash_constants(
                    vocab_size=self._config.vocab_size,
                    ngram_size=args.ngram_size,
                    num_ngram_heads=args.num_ngram_heads,
                    ngram_vocab_size_base=args.ngram_vocab_size_base,
                    ple_layer_index=ple.ple_index,
                )
                emb = ple.ple_embedding
                emb.layer_multipliers.copy_(torch.tensor(mult, dtype=torch.int64))
                emb.ngram_heads_vocab_sizes.copy_(torch.tensor(sizes, dtype=torch.int64))
                emb.ngram_heads_offsets.copy_(torch.tensor(offsets, dtype=torch.int64))
                emb.attach_table(ZeroTable(offsets[-1] + sizes[-1], args.ngram_head_dim))
            return 0

        from freetoken.models.gguf.reader import is_gguf_path

        is_gguf = is_gguf_path(engine_config.model_path)

        if engine_config.ple_backend == "disk":
            from freetoken.utils import download_hf_weight

            from .ple_disk import DiskRowTable, resolve_row_source

            folder = (
                engine_config.model_path
                if is_gguf
                else download_hf_weight(engine_config.model_path)
            )
            # one WAIT node per captured graph: the flag protocol supports a single consume
            assert len(ple_layers) == 1, "disk PLE backend expects exactly one PLE layer"
            emb, args = ple_layers[0].ple_embedding, ple_layers[0].args
            # hash with the state-dict-loaded constants, the same source the pinned path reads
            constants = {
                "num_ngram_heads": args.num_ngram_heads,
                "layer_multipliers": emb.layer_multipliers.tolist(),
                "per_head_vocab_sizes": emb.ngram_heads_vocab_sizes.tolist(),
                "per_head_offsets": emb.ngram_heads_offsets.tolist(),
                "eos_token_id": args.ngram_boundary_token_id,
                "image_token_id": args.image_token_id,
                "ngram_head_dim": args.ngram_head_dim,
            }
            disk_table = DiskRowTable(
                resolve_row_source(folder),
                constants,
                max_graph_rows=max(256, engine_config.cuda_graph_max_bs or 0),
                max_extend_tokens=engine_config.max_extend_tokens,
            )
            self._ple_table = disk_table
            for ple in ple_layers:
                ple.ple_embedding.attach_table(disk_table)
            # engine enters this around every dispatch; the graph itself never waits on the disk
            self.forward_host_ctx = disk_table.forward_host_ctx
            return 0

        if is_gguf:
            from .gguf import load_ple_table_from_gguf

            table = load_ple_table_from_gguf(engine_config.model_path, self._config.qwen4_args)
            self._ple_table = table
            for ple in ple_layers:
                ple.ple_embedding.attach_table(table)
            return 0

        from .weight import load_ple_table

        table = load_ple_table(engine_config.model_path, self._config.qwen4_args)
        self._ple_table = table  # owns the pinned HostBank; keep it alive
        for ple in ple_layers:
            ple.ple_embedding.attach_table(
                PinnedUVATable(table.bank.tensor, float(table.weight_scale))
            )
        return table.bank.nbytes

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        return self.lm_head.forward(self.model.forward(batch.input_ids, batch))


class Qwen4ExpForConditionalGeneration(QwenVLVisionMixin, Qwen4ExpForCausalLM):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if config.is_multimodal:
            assert not config.vision_config.deepstack_visual_indexes, (
                "Qwen3.8 consumes no DeepStack features"
            )
            self.visual = Qwen3VLVisionModel(
                config.vision_config, quant_config=config.quant, prefix="visual"
            )


__all__ = [
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpModel",
    "Qwen4ExpMTP",
    "build_linear_mixer",
]
