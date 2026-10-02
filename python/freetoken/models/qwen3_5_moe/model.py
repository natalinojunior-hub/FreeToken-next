from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    OPList,
    LinearReplicated,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.blocks import embed_input_ids
from freetoken.models.qwen3_vl.vision import Qwen3VLVisionModel, QwenVLVisionMixin
from freetoken.utils import nvtx_annotate

from .attention import Qwen3_5Attention
from .gdn import Qwen3_5GatedDeltaNet
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen3_5DecoderLayer(BaseOP):
    """Pre-norm hybrid block: ``x = x + mixer(input_norm(x)); x = x + moe(post_norm(x))``,
    where the mixer is a GatedDeltaNet (linear layers) or gated attention (full layers).
    All norms are Gemma-style (1+weight)."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                quant_config=config.quant,
                prefix=f"{prefix}.linear_attn",
            )
        else:
            self.self_attn = Qwen3_5Attention(config, layer_id, prefix=f"{prefix}.self_attn")
        # Dense variants (num_experts==0, e.g. Qwen3.6-27B) use a plain SwiGLU MLP instead of
        # the routed MoE block; both expose ``forward(hidden)->hidden`` and the same key prefix.
        self.mlp = (
            Qwen3_5MoE(config, layer_id, prefix=f"{prefix}.mlp")
            if config.moe_enabled
            else Qwen3_5DenseMLP(config, prefix=f"{prefix}.mlp")
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        # Residual-stream form: fuse each residual-add into the next RMSNorm
        # (GemmaRMSNorm.forward_add_residual) so add + norm are one kernel per sublayer.
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        hidden = (
            self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
        )
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [
                Qwen3_5DecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._capture_mtp_residual = config.mtp_layer_id is not None
        self._last_residual: torch.Tensor | None = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = embed_input_ids(self.embed_tokens, input_ids, get_global_ctx().batch)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        x, _ = self.norm.forward_add_residual(x, residual)
        if self._capture_mtp_residual:
            self._last_residual = x  # the NextN draft seeds from the post-norm hidden
        return x


class Qwen3_5MTP(BaseOP):
    """The checkpoint's NextN draft block (llama.cpp ``graph_mtp`` for qwen35/qwen35moe):
    ``eh_proj(enorm(embed(next)) ++ hnorm(h))`` -> one full-attention decoder layer ->
    ``shared_head_norm``. ``forward`` returns that normed hidden, which both seeds the next
    draft step and feeds the shared LM head (``to_head`` is the identity)."""

    prime_kv_without_experts = True

    def __init__(self, config: ModelConfig, layer_id: int, *, embedding: BaseOP) -> None:
        if config.mtp_layer_id != layer_id or layer_id != config.num_layers:
            raise ValueError("register the MTP layer with with_mtp_layer before construction")
        hidden = config.hidden_size
        self._embed_ref = embedding
        self.enorm = GemmaRMSNorm(hidden, eps=config.rms_norm_eps)
        self.hnorm = GemmaRMSNorm(hidden, eps=config.rms_norm_eps)
        self.eh_proj = LinearReplicated(
            2 * hidden, hidden, has_bias=False, quant_config=config.quant, prefix="mtp.eh_proj"
        )
        # Standalone HF heads store their own BF16 experts. Keep them independent of the
        # target's quantized/offloaded expert bank; the model memory planner prices these
        # resident parameters from their actual shapes.
        mtp_config = (
            replace(config, quant=None, moe_strategy="fused")
            if config.mtp_expert_resident
            else config
        )
        self.layers = OPList([Qwen3_5DecoderLayer(mtp_config, layer_id, prefix="mtp.layers.0")])
        self.shared_head_norm = GemmaRMSNorm(hidden, eps=config.rms_norm_eps)

    def forward(self, residual: torch.Tensor, next_ids: torch.Tensor, batch) -> torch.Tensor:
        x = self._prepare_hidden(residual, next_ids)
        x, res = self.layers.op_list[0].forward(x, None)
        x, _ = self.shared_head_norm.forward_add_residual(x, res)
        return x

    def _prepare_hidden(self, residual: torch.Tensor, next_ids: torch.Tensor) -> torch.Tensor:
        e = self.enorm.forward(self._embed_ref.forward(next_ids).to(residual.dtype))
        return self.eh_proj.forward(torch.cat([e, self.hnorm.forward(residual)], dim=-1))

    def prime_kv(self, residual: torch.Tensor, next_ids: torch.Tensor, batch) -> None:
        """Store target-fed draft KV without unused attention outputs or expert work."""
        layer = self.layers.op_list[0]
        hidden = layer.input_layernorm.forward(self._prepare_hidden(residual, next_ids))
        _, k, v, _ = layer.self_attn._project(hidden)
        get_global_ctx().kv_cache.store_kv(k, v, batch.out_loc, layer.self_attn.layer_id)

    def to_head(self, residual: torch.Tensor) -> torch.Tensor:
        return residual


class Qwen3_5ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        self.mtp = (
            Qwen3_5MTP(config, config.mtp_layer_id, embedding=self.model.embed_tokens)
            if config.mtp_layer_id is not None
            else None
        )
        super().__init__()

        # A GGUF checkpoint carries native block-quantized weights: swap the dense
        # projections + embedding for GGUF-quant ops so the packed buffers have somewhere
        # to land (routed experts stay on the offload cache). Mirrors gemma4/model.py.
        from .gguf import convert_qwen35_to_gguf, is_gguf_model

        if is_gguf_model(config):
            assert config.gguf_model_path is not None, (
                "expert_quant=='gguf' but ModelConfig.gguf_model_path is unset; the per-tensor "
                "ggml types can only be read from the file"
            )
            convert_qwen35_to_gguf(self, config, model_path=config.gguf_model_path)

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
    """The MoE releases share the dense code path: the decoder picks the routed or dense MLP from config.num_experts."""


class Qwen3_5ForConditionalGeneration(QwenVLVisionMixin, Qwen3_5ForCausalLM):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.is_multimodal:
            assert not config.vision_config.deepstack_visual_indexes, (
                "Qwen3.5 consumes no DeepStack features"
            )
            self.visual = Qwen3VLVisionModel(
                config.vision_config, quant_config=config.quant, prefix="visual"
            )


class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """The MoE releases with the vision tower; see Qwen3_5MoeForCausalLM."""


__all__ = [
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
]
