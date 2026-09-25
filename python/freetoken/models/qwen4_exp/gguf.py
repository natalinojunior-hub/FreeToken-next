"""GGUF adapter for Qwen3.8-Flash-Next (qwen4exp).

Parses GGUF metadata produced by llama.cpp / unsloth into FreeToken ModelConfig,
and translates GGUF tensor names/layouts into FreeToken's qwen4_exp weight map.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)
from freetoken.models.qwen3_5_moe.gguf_experts import (
    gguf_expert_types as _gguf_expert_types_base,
    load_gguf_expert_sources as _load_gguf_expert_sources_base,
)
from freetoken.models.qwen3_5_moe.gguf import _ungroup_packed_rows, _ungroup_v
from freetoken.models.qwen4_exp.config import Qwen4ExpArgs, Qwen4ExpMTPConfig, ple_slot_states

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim


def _kv(shim: "GgufConfigShim", suffix: str, default=None):
    arch = shim.model_type
    key = f"{arch}.{suffix}"
    if key in shim.metadata:
        return shim.metadata[key]
    key_gen = f"general.{suffix}"
    if key_gen in shim.metadata:
        return shim.metadata[key_gen]
    if default is not None:
        return default
    raise KeyError(f"GGUF metadata missing key {key!r} (and {key_gen!r})")


MTP_PATH_ENV = "FREETOKEN_MTP_PATH"


def _find_mtp_gguf_path(model_path: str) -> str | None:
    """The MTP head GGUF: ``--mtp`` / ``FREETOKEN_MTP_PATH``, else ``<model>-mtp.gguf`` next to
    the main model, else the first GGUF in an ``MTP/`` dir beside it or its parent."""
    from freetoken.models.gguf.reader import gguf_companion_path, resolve_gguf_path

    if override := os.environ.get(MTP_PATH_ENV):
        if not os.path.isfile(override):
            raise FileNotFoundError(f"--mtp {override!r} does not exist")
        return override
    if found := gguf_companion_path(model_path, "mtp"):
        return found
    main_path = resolve_gguf_path(model_path)
    if main_path is None:
        return None
    main_dir = os.path.dirname(main_path)
    for mtp_dir in (os.path.join(main_dir, "MTP"), os.path.join(os.path.dirname(main_dir), "MTP")):
        if os.path.isdir(mtp_dir):
            files = sorted(
                f for f in os.listdir(mtp_dir) if f.endswith(".gguf") and not f.startswith(".")
            )
            if files:
                return os.path.join(mtp_dir, files[0])
    return None


def _parse_mtp_config_from_gguf(model_path: str) -> Qwen4ExpMTPConfig:
    """Parse MTP config from MTP GGUF metadata."""
    mtp_path = _find_mtp_gguf_path(model_path)
    if mtp_path is None:
        return Qwen4ExpMTPConfig()
    from freetoken.models.gguf.config import build_gguf_shim

    shim = build_gguf_shim(mtp_path)
    enabled = shim.metadata.get("qwen4exp.nextn_predict_layers", 0) > 0
    hybrid = True
    num_hidden_layers = shim.metadata.get("qwen4exp.nextn_predict_layers", 0)
    layer_types = ("full_attention",)
    # nextn_shared_target_tensors covers the embedding / LM head the file omits; the draft
    # block's routed experts are its own (blk.{N}.ffn_*_exps, independent of any target layer).
    types = None
    if enabled:
        layer = _mtp_gguf_layer(shim)
        t = _gguf_expert_types_base(mtp_path, 0, ((mtp_path, layer),))
        types = (t["gate_up"][0], t["down"][0])
    return Qwen4ExpMTPConfig(
        enabled=enabled,
        hybrid=hybrid,
        num_hidden_layers=num_hidden_layers,
        layer_types=layer_types,
        use_hidden_state_from_layer=None,
        rope_theta=None,
        gguf_expert_types=types,
    )


def _mtp_gguf_layer(mtp_shim) -> int:
    """GGUF block index of the (single) draft block: it follows the target stack."""
    return int(mtp_shim.metadata["qwen4exp.block_count"]) - int(
        mtp_shim.metadata["qwen4exp.nextn_predict_layers"]
    )


def _mtp_extra_banks(
    model_path: str, num_banks: int, num_target: int
) -> tuple[tuple[str, int], ...]:
    """``(mtp_gguf_path, gguf_layer)`` for the banks past the target's: the draft's own."""
    if num_banks == num_target:
        return ()
    mtp_path = _find_mtp_gguf_path(model_path)
    if num_banks != num_target + 1 or mtp_path is None:
        raise ValueError(
            f"{num_banks} expert banks requested for a {num_target}-layer target; only one "
            f"extra (the MTP draft's, from its MTP/*.gguf) is supported"
        )
    from freetoken.models.gguf.config import build_gguf_shim

    return ((mtp_path, _mtp_gguf_layer(build_gguf_shim(mtp_path))),)


def gguf_expert_types(model_path: str, num_layers: int) -> dict[str, list[int]]:
    """Per-bank expert ggml types; ``num_layers`` past the target's block count includes
    the MTP draft's own bank (``ModelConfig.num_moe_layers`` with ``mtp_expert_bank``)."""
    from freetoken.models.gguf.config import build_gguf_shim
    from freetoken.models.gguf.reader import resolve_gguf_path

    num_target = int(_kv(build_gguf_shim(resolve_gguf_path(model_path)), "block_count"))
    extra = _mtp_extra_banks(model_path, max(num_layers, num_target), num_target)
    return _gguf_expert_types_base(model_path, min(num_layers, num_target), extra)


def load_gguf_expert_sources(
    model_path: str, config: ModelConfig, *, layer_sink=None
) -> dict[str, list[torch.Tensor]]:
    """Target expert banks plus, with ``config.mtp_expert_bank``, the MTP draft's own."""
    extra = _mtp_extra_banks(model_path, config.num_moe_layers, config.num_layers)
    return _load_gguf_expert_sources_base(
        model_path, config, layer_sink=layer_sink, extra_banks=extra
    )


def is_gguf_model(config: ModelConfig) -> bool:
    return getattr(config, "gguf_model_path", None) is not None


def _scan_quant_types(model_path: str) -> dict[tuple[int, str], int]:
    from freetoken.models.gguf.reader import iter_gguf_tensors

    quant_types = {}
    for t in iter_gguf_tensors(model_path, skip_names={"per_layer_token_embd.weight"}):
        if not t.name.startswith("blk."):
            quant_types[(-1, t.name)] = t.ggml_type
            continue
        _, idx, suffix = t.name.split(".", 2)
        quant_types[(int(idx), suffix)] = t.ggml_type
    return quant_types


def _dense(t, dtype: torch.dtype, device=None) -> torch.Tensor:
    from freetoken.models.gguf.dequant import (
        BLOCK_SHAPE,
        GGML_BF16,
        GGML_F16,
        GGML_F32,
        GGML_UNQUANTIZED,
    )

    gt = int(t.ggml_type)
    raw = t.packed()
    if gt in GGML_UNQUANTIZED:
        view = {GGML_F32: torch.float32, GGML_F16: torch.float16, GGML_BF16: torch.bfloat16}[gt]
        res = raw.reshape(-1).view(view).reshape(t.shape).to(dtype)
        return res if device is None else res.to(device)

    from freetoken.kernel.gguf import ggml_dequantize

    block, type_size = BLOCK_SHAPE[gt]
    in_features = t.row_bytes // type_size * block
    cuda_dev = (
        device
        if device is not None and getattr(device, "type", "") == "cuda"
        else torch.device("cuda:0")
    )
    out = ggml_dequantize(raw.to(cuda_dev).contiguous(), gt, t.rows, in_features, torch.bfloat16)
    res = out.reshape(t.shape).to(dtype)
    return res if device is None else res.to(device)


def _to_bf16(t, device=None) -> torch.Tensor:
    return _dense(t, torch.bfloat16, device=device)


def _to_f32(t, device=None) -> torch.Tensor:
    return _dense(t, torch.float32, device=device)


def _plus_one_norm(t, device=None) -> torch.Tensor:
    """A (1+w) RMSNorm weight, back to the checkpoint's zero-centered ``w``.

    llama.cpp's converter folds the +1 into every Qwen4ExpTextRMSNorm tensor it writes
    (measured: GGUF - HF == +1.0000 exactly, std 0, for hc/ple/q/k/indexer/MTP norms),
    while GroupedPlusOneRMSNorm / GemmaPlusOneRMSNorm apply (1+w) at runtime -- loading the
    GGUF value raw scales every norm by 2+w. ``ssm_norm`` (GatedRMSNorm, plain w) is not shifted.
    """
    return (_to_f32(t, device=device) - 1.0).to(torch.bfloat16)


def _ungroup_packed_cols(t, num_k_heads: int, num_v_per_k: int, head_dim: int) -> torch.Tensor:
    """Un-tile the V-head COLUMNS of a packed ``ssm_out`` without dequantizing.

    qwen3_5_moe's loader densifies this tensor because a 128-wide head straddles 256-element
    k-quant blocks; a head that is a whole number of blocks permutes as raw bytes instead.
    """
    from freetoken.models.gguf.dequant import BLOCK_SHAPE

    block, type_size = BLOCK_SHAPE[int(t.ggml_type)]
    if head_dim % block:
        raise NotImplementedError(
            f"{t.name}: V head of {head_dim} columns straddles {block}-element "
            f"ggml type {t.ggml_type} blocks; use _tiled_input_perm instead"
        )
    head_bytes = head_dim // block * type_size
    packed = t.packed()
    rows = packed.reshape(t.rows, -1)
    return _ungroup_v(rows, 1, num_k_heads, num_v_per_k, head_bytes).reshape(packed.shape)


def _straddles_blocks(ggml_type: int, head_dim: int) -> bool:
    """A V head is not a whole number of ``ggml_type`` blocks (e.g. 128 in 256-wide K/I-quants)."""
    from freetoken.models.gguf.dequant import BLOCK_SHAPE

    return head_dim % BLOCK_SHAPE[int(ggml_type)][0] != 0


def _tiled_input_perm(num_k_heads: int, num_v_per_k: int, head_dim: int) -> torch.Tensor:
    """Index that reorders a grouped V activation into llama.cpp's tiled column order.

    When the packed ``ssm_out`` cannot be un-tiled in place (``_straddles_blocks``), the weight
    stays tiled and its input is permuted instead: ``W_tiled @ x[perm] == W_grouped @ x``.
    """
    n = num_k_heads * num_v_per_k * head_dim
    grouped_to_tiled = _ungroup_v(torch.arange(n), 0, num_k_heads, num_v_per_k, head_dim)
    return torch.argsort(grouped_to_tiled)


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    """Parse qwen4exp GGUF metadata into ModelConfig."""
    num_layers = int(_kv(shim, "block_count"))
    hidden_size = int(_kv(shim, "embedding_length"))
    num_qo_heads = int(_kv(shim, "attention.head_count"))
    num_kv_heads = int(_kv(shim, "attention.head_count_kv"))
    head_dim = int(_kv(shim, "attention.key_length"))
    rms_eps = float(_kv(shim, "attention.layer_norm_rms_epsilon"))
    rope_base = float(_kv(shim, "rope.freq_base"))
    rotary_dim = int(_kv(shim, "rope.dimension_count"))
    max_pos = int(_kv(shim, "context_length"))

    num_experts = int(_kv(shim, "expert_count", 0))
    experts_per_tok = int(_kv(shim, "expert_used_count", 0))
    moe_inter = int(_kv(shim, "expert_feed_forward_length", 0))
    shared_inter = int(_kv(shim, "expert_shared_feed_forward_length", 0))
    moe_enabled = num_experts > 0

    conv_kernel = int(_kv(shim, "ssm.conv_kernel"))
    state_size = int(_kv(shim, "ssm.state_size"))
    num_k_heads = int(_kv(shim, "ssm.group_count"))
    num_v_heads = int(_kv(shim, "ssm.time_step_rank"))

    interval = int(_kv(shim, "full_attention_interval", 4))
    full_ids = tuple(i for i in range(num_layers) if (i + 1) % interval == 0)
    linear_ids = tuple(i for i in range(num_layers) if i not in set(full_ids))

    # With a vision tower (mmproj), image tokens need 3-axis positions: the checkpoint's
    # interleaved mrope sections (llama.cpp "imrope"), exactly as the HF path configures them.
    # Text-only serving keeps plain rope (identical for 1-axis positions).
    sections = shim.metadata.get("qwen4exp.rope.dimension_sections")
    mrope = getattr(shim, "vision_config", None) is not None and sections is not None
    full_rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        max_position=max_pos,
        base=rope_base,
        scaling=None,
        mrope_section=[int(x) for x in sections[:3]] if mrope else None,
        mrope_layout="interleaved" if mrope else "contiguous",
    )

    groups = (
        FullAttentionGroupConfig(
            name="full",
            layer_ids=full_ids,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_config=full_rotary,
            index_head_dim=int(_kv(shim, "attention.indexer.key_length", 128)),
            num_index_layers=len(full_ids),
            index_ratio=int(_kv(shim, "attention.compress_ratios", [4])[interval - 1] or 4),
        ),
        LinearGatedDeltaGroupConfig(
            name="linear",
            layer_ids=linear_ids,
            num_key_heads=num_k_heads,
            num_value_heads=num_v_heads,
            key_head_dim=state_size,
            value_head_dim=state_size,
            conv_kernel_dim=conv_kernel,
            # HF config.json output_gate_type is "sigmoid"; the GGUF carries no such key
            output_gate=str(_kv(shim, "ssm.output_gate_type", "sigmoid")),
        ),
    )

    # Qwen4Exp specific extensions (PLE, Hyper-Connections, MTP)
    ngram_size = int(_kv(shim, "ple.ngram_size", 3))
    heads_per_ngram = int(_kv(shim, "ple.heads_per_ngram", 8))
    ple_layers = tuple(int(x) for x in _kv(shim, "ple.layers", [1]))
    hc_count = int(_kv(shim, "hyper_connection.count", 4))
    hc_rank = int(_kv(shim, "hyper_connection.low_rank", 320))
    ple_conv = int(_kv(shim, "ple.conv_kernel", 4))
    eos_id = int(_kv(shim, "ple.eos_token_id", 248044))
    img_id = int(_kv(shim, "ple.image_token_id", 248056))

    # Detect ple_embed_dim from GGUF tensor shapes (metadata may be wrong)
    ple_dim = int(_kv(shim, "embedding_length_per_layer_input", 160))
    model_path = getattr(shim, "model_path", None)
    if model_path is not None:
        from freetoken.models.gguf.reader import iter_gguf_tensors

        for t in iter_gguf_tensors(model_path, skip_names={"per_layer_token_embd.weight"}):
            if t.name.endswith(".ple_key.weight"):
                # ple_key.weight shape is [out_features, in_features] = [width, ple_embed_dim]
                ple_dim = t.shape[1]
                break

    mtp_config = _parse_mtp_config_from_gguf(model_path) if model_path else Qwen4ExpMTPConfig()

    qwen4_args = Qwen4ExpArgs(
        hidden_size=hidden_size,
        hc_count=hc_count,
        hc_lowrank=hc_rank,
        ple_layer_ids=ple_layers,
        ple_embed_dim=ple_dim,
        ple_conv_kernel_size=ple_conv,
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        ngram_vocab_size_base=20000000,
        make_ngram_vocab_size_divisible_by=64,
        split_ngram_parts=1,
        ngram_boundary_token_id=eos_id,
        index_n_heads=int(_kv(shim, "attention.indexer.head_count", 4)),
        index_kv_heads=int(_kv(shim, "attention.indexer.head_count_kv", 1)),
        index_head_dim=int(_kv(shim, "attention.indexer.key_length", 128)),
        index_budget=int(_kv(shim, "attention.indexer.top_k", 2048)),
        index_ratio=int(_kv(shim, "attention.compress_ratios", [4])[interval - 1] or 4),
        image_token_id=img_id,
        mtp=mtp_config,
    )

    model_path = getattr(shim, "model_path", None)
    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        vocab_size=shim.vocab_size,
        intermediate_size=moe_inter,
        hidden_act="silu",
        rms_norm_eps=rms_eps,
        norm_topk_prob=True,
        tie_word_embeddings=shim.tie_word_embeddings,
        rotary_config=full_rotary,
        num_experts=num_experts,
        num_experts_per_tok=experts_per_tok,
        moe_intermediate_size=moe_inter,
        shared_expert_intermediate_size=shared_inter,
        moe_enabled=moe_enabled,
        use_qk_norm=True,
        model_type="qwen4_exp",
        architectures=list(shim.architectures),
        attention_groups=groups,
        qwen4_args=qwen4_args,
        expert_quant="gguf" if moe_enabled else "none",
        gguf_expert_types=(gguf_expert_types(model_path, num_layers) if model_path else None),
        gguf_model_path=model_path,
        vision_config=getattr(shim, "vision_config", None),
        image_token_id=getattr(shim, "image_token_id", None),
        slot_states=ple_slot_states(qwen4_args),
        attn_quant="gguf",
        dense_quant="gguf",
        lm_head_quant="gguf",
        kv_ram_tier_certified=True,
    )


def convert_qwen4exp_to_gguf(model, config: ModelConfig, *, model_path: str) -> None:
    """In-place: replace qwen4_exp's dense projections + embedding with native GGUF ops."""
    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear, gguf_merged_or_plain

    quant_map = _scan_quant_types(model_path)
    full_layer_ids = {
        lid
        for lid in range(config.num_layers)
        if isinstance(config.attention_group_for_layer(lid), FullAttentionGroupConfig)
    }

    _qkv_split = [
        config.num_qo_heads * config.head_dim * 2,
        config.num_kv_heads * config.head_dim,
        config.num_kv_heads * config.head_dim,
    ]
    gdn_group = next(
        group for group in config.attention_groups if isinstance(group, LinearGatedDeltaGroupConfig)
    )
    _in_proj_split = [
        2 * gdn_group.num_key_heads * gdn_group.key_head_dim
        + gdn_group.num_value_heads * gdn_group.value_head_dim,
        gdn_group.num_value_heads * gdn_group.value_head_dim,
        gdn_group.num_value_heads,
        gdn_group.num_value_heads,
    ]
    _index_split = [
        config.qwen4_args.index_n_heads * config.qwen4_args.index_head_dim,
        config.qwen4_args.index_kv_heads * config.qwen4_args.index_head_dim,
    ]

    def qt(layer: int, suffix: str) -> int:
        return quant_map.get((layer, suffix), 0)

    def swap_linear(owner, attr: str, quant_type: int) -> None:
        lin = getattr(owner, attr)
        out_features, in_features = lin.weight.shape
        setattr(
            owner,
            attr,
            GGUFLinear(in_features, out_features, quant_type, has_bias=lin.bias is not None),
        )

    inner = model.model
    inner.embed_tokens = GGUFEmbedding(
        num_embeddings=config.vocab_size,
        embedding_dim=config.hidden_size,
        quant_type=qt(-1, "token_embd.weight"),
    )
    # The MTP head shares the token embedding by reference; re-point it at the
    # GGUF module, else it keeps the replaced (never-loaded, meta) embedding.
    if getattr(model, "mtp", None) is not None:
        model.mtp._embed_ref = inner.embed_tokens
    if not config.tie_word_embeddings and hasattr(model, "lm_head"):
        from freetoken.layers.gguf import GGUFLMHead

        head = model.lm_head
        out_features, in_features = head.weight.shape
        model.lm_head = GGUFLMHead(
            in_features,
            out_features,
            qt(-1, "output.weight"),
            has_bias=head.bias is not None,
        )

    hc_top = inner.hyper_connection_mixer
    swap_linear(hc_top, "input_mix_weight_up", qt(-1, "output_hc_up.weight"))
    swap_linear(hc_top, "input_mix_weight_down", qt(-1, "output_hc_down.weight"))

    for layer_id, layer in enumerate(inner.layers.op_list):
        swap_linear(
            layer.attn_hyper_connection, "input_mix_weight_up", qt(layer_id, "hc_attn_up.weight")
        )
        swap_linear(
            layer.mlp_hyper_connection, "input_mix_weight_up", qt(layer_id, "hc_ffn_up.weight")
        )

        if layer_id in full_layer_ids:
            layer.self_attn.qkv_proj = gguf_merged_or_plain(
                config.hidden_size,
                _qkv_split,
                [
                    qt(layer_id, "attn_q.weight"),
                    qt(layer_id, "attn_k.weight"),
                    qt(layer_id, "attn_v.weight"),
                ],
            )
            swap_linear(layer.self_attn, "o_proj", qt(layer_id, "attn_output.weight"))
            layer.self_attn.indexer.index_qk_proj = gguf_merged_or_plain(
                config.hidden_size,
                _index_split,
                [
                    qt(layer_id, "indexer.q_proj.weight"),
                    qt(layer_id, "indexer.k_proj.weight"),
                ],
            )
        else:
            layer.linear_attn.in_proj = gguf_merged_or_plain(
                config.hidden_size,
                _in_proj_split,
                [
                    qt(layer_id, "attn_qkv.weight"),
                    qt(layer_id, "attn_gate.weight"),
                    qt(layer_id, "ssm_beta.weight"),
                    qt(layer_id, "ssm_alpha.weight"),
                ],
            )
            swap_linear(layer.linear_attn, "out_proj", qt(layer_id, "ssm_out.weight"))
            gdn = layer.linear_attn
            if gdn.num_k_heads != gdn.num_v_heads and _straddles_blocks(
                qt(layer_id, "ssm_out.weight"), gdn.head_v_dim
            ):
                # iter_gguf_weights keeps this out_proj tiled and supplies the input order
                gdn.out_proj_in_perm = torch.empty(gdn.value_dim, dtype=torch.int64)

        layer.mlp.shared_expert.gate_up_proj = gguf_merged_or_plain(
            config.hidden_size,
            [config.shared_expert_intermediate_size, config.shared_expert_intermediate_size],
            [
                qt(layer_id, "ffn_gate_shexp.weight"),
                qt(layer_id, "ffn_up_shexp.weight"),
            ],
        )
        swap_linear(layer.mlp.shared_expert, "down_proj", qt(layer_id, "ffn_down_shexp.weight"))

        if layer.ple is not None:
            swap_linear(layer.ple, "key_proj", qt(layer_id, "ple_key.weight"))
            swap_linear(layer.ple, "value_proj", qt(layer_id, "ple_value.weight"))

    # Initialize PLE embedding constants (layer_multipliers, ngram_heads_vocab_sizes, ngram_heads_offsets)
    # These are derived from hash constants and must match the HF checkpoint values for state dict loading.
    if config.qwen4_args is not None:
        from .ple import derive_ngram_hash_constants

        qwen4_args = config.qwen4_args
        for ple_layer_id in qwen4_args.ple_layer_ids:
            ple_layer = inner.layers.op_list[ple_layer_id]
            if ple_layer.ple is not None:
                emb = ple_layer.ple.ple_embedding
                mult, sizes, offsets = derive_ngram_hash_constants(
                    vocab_size=config.vocab_size,
                    ngram_size=qwen4_args.ngram_size,
                    num_ngram_heads=qwen4_args.num_ngram_heads,
                    ngram_vocab_size_base=qwen4_args.ngram_vocab_size_base,
                    ple_layer_index=ple_layer.ple.ple_index,
                )
                emb.layer_multipliers.copy_(torch.tensor(mult, dtype=torch.int64))
                emb.ngram_heads_vocab_sizes.copy_(torch.tensor(sizes, dtype=torch.int64))
                emb.ngram_heads_offsets.copy_(torch.tensor(offsets, dtype=torch.int64))


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool = False,
    include_non_moe: bool = True,
    include_vision: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Iterate and yield weights from GGUF for qwen4_exp."""
    from freetoken.models.gguf.reader import iter_gguf_tensors, resolve_gguf_path
    from freetoken.models.qwen4_exp.gguf import parse_gguf_config
    from freetoken.models.gguf.config import build_gguf_shim

    assert include_non_moe
    quant_map = _scan_quant_types(model_path)

    # Parse config to get Qwen4ExpArgs for PLE embedding weights
    shard1_path = resolve_gguf_path(model_path)
    if shard1_path is None:
        raise ValueError(f"Cannot resolve GGUF path: {model_path}")
    shim = build_gguf_shim(shard1_path)
    config = parse_gguf_config(shim)
    qwen4_args = config.qwen4_args
    # llama.cpp stores GDN V heads TILED when K heads < V heads; FreeToken wants them grouped
    # (see qwen3_5_moe/gguf.py _ungroup_v). Every tensor indexing the V-head axis is un-tiled.
    gdn = next(g for g in config.attention_groups if isinstance(g, LinearGatedDeltaGroupConfig))
    vK, vD = gdn.num_key_heads, gdn.value_head_dim
    vR = gdn.num_value_heads // vK
    untile = vK != gdn.num_value_heads
    qk_rows = 2 * vK * gdn.key_head_dim

    qkv_buf: dict[int, dict[str, torch.Tensor]] = {}
    index_buf: dict[int, dict[str, torch.Tensor]] = {}
    in_proj_buf: dict[int, dict[str, torch.Tensor]] = {}
    gate_up_buf: dict[int, dict[str, torch.Tensor]] = {}
    hc_attn_down_buf: dict[int, dict[str, torch.Tensor]] = {}
    hc_ffn_down_buf: dict[int, dict[str, torch.Tensor]] = {}

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "per_layer_token_embd.weight":
            continue
        if not include_moe_experts and (
            ".expert." in name
            or "ffn_gate_exps" in name
            or "ffn_down_exps" in name
            or "ffn_up_exps" in name
        ):
            continue

        if not name.startswith("blk."):
            if name == "token_embd.weight":
                yield "model.embed_tokens.qweight", t.packed()
            elif name == "output.weight":
                yield "lm_head.qweight", t.packed()
            elif name == "output_hc_norm.weight":
                yield "model.hyper_connection_mixer.hc_norm.weight", _plus_one_norm(t)
            elif name == "output_hc_up.weight":
                yield "model.hyper_connection_mixer.input_mix_weight_up.qweight", t.packed()
            elif name == "output_hc_down.weight":
                yield "model.hyper_connection_mixer.input_mix_weight_down.qweight", t.packed()
            continue

        parts = name.split(".", 2)
        layer = int(parts[1])
        suffix = parts[2]
        base = f"model.layers.{layer}"

        # HC norm & up projections
        if suffix == "hc_attn_norm.weight":
            yield f"{base}.attn_hyper_connection.hc_norm.weight", _plus_one_norm(t)
            continue
        if suffix == "hc_ffn_norm.weight":
            yield f"{base}.mlp_hyper_connection.hc_norm.weight", _plus_one_norm(t)
            continue
        if suffix == "hc_attn_up.weight":
            yield f"{base}.attn_hyper_connection.input_mix_weight_up.qweight", t.packed()
            continue
        if suffix == "hc_ffn_up.weight":
            yield f"{base}.mlp_hyper_connection.input_mix_weight_up.qweight", t.packed()
            continue

        # HC down + inject fusion (into BF16 LinearReplicated)
        if suffix == "hc_attn_down.weight":
            hc_attn_down_buf.setdefault(layer, {})["down"] = _to_bf16(t, device=device)
        elif suffix == "hc_attn_inject.weight":
            hc_attn_down_buf.setdefault(layer, {})["inject"] = _to_bf16(t, device=device)
        if (
            layer in hc_attn_down_buf
            and "down" in hc_attn_down_buf[layer]
            and "inject" in hc_attn_down_buf[layer]
        ):
            d = hc_attn_down_buf[layer]["down"].to(device)
            inj = hc_attn_down_buf[layer]["inject"].to(device)
            pad = torch.zeros(12, d.shape[1], dtype=d.dtype, device=device)
            yield (
                f"{base}.attn_hyper_connection.input_mix_weight_down_block_inject.weight",
                torch.cat([d, inj, pad], dim=0),
            )
            del hc_attn_down_buf[layer]
            continue

        if suffix == "hc_ffn_down.weight":
            hc_ffn_down_buf.setdefault(layer, {})["down"] = _to_bf16(t, device=device)
        elif suffix == "hc_ffn_inject.weight":
            hc_ffn_down_buf.setdefault(layer, {})["inject"] = _to_bf16(t, device=device)
        if (
            layer in hc_ffn_down_buf
            and "down" in hc_ffn_down_buf[layer]
            and "inject" in hc_ffn_down_buf[layer]
        ):
            d = hc_ffn_down_buf[layer]["down"].to(device)
            inj = hc_ffn_down_buf[layer]["inject"].to(device)
            pad = torch.zeros(12, d.shape[1], dtype=d.dtype, device=device)
            yield (
                f"{base}.mlp_hyper_connection.input_mix_weight_down_block_inject.weight",
                torch.cat([d, inj, pad], dim=0),
            )
            del hc_ffn_down_buf[layer]
            continue

        # MoE router & shared expert
        if suffix == "ffn_gate_inp.weight":
            yield f"{base}.mlp.gate.weight", _to_bf16(t)
            continue
        if suffix == "ffn_gate_inp_shexp.weight":
            yield f"{base}.mlp.shared_expert_gate.weight", _to_bf16(t).reshape(1, -1)
            continue
        if suffix == "ffn_down_shexp.weight":
            yield f"{base}.mlp.shared_expert.down_proj.qweight", t.packed()
            continue
        if suffix == "ffn_gate_shexp.weight":
            gate_up_buf.setdefault(layer, {})["gate"] = t.packed()
        elif suffix == "ffn_up_shexp.weight":
            gate_up_buf.setdefault(layer, {})["up"] = t.packed()
        if layer in gate_up_buf and "gate" in gate_up_buf[layer] and "up" in gate_up_buf[layer]:
            gu = gate_up_buf[layer]
            t_gate = quant_map.get((layer, "ffn_gate_shexp.weight"))
            t_up = quant_map.get((layer, "ffn_up_shexp.weight"))
            if t_gate == t_up:
                yield (
                    f"{base}.mlp.shared_expert.gate_up_proj.qweight",
                    torch.cat([gu["gate"], gu["up"]], dim=0),
                )
            else:
                yield f"{base}.mlp.shared_expert.gate_up_proj.qweight_0", gu["gate"]
                yield f"{base}.mlp.shared_expert.gate_up_proj.qweight_1", gu["up"]
            del gate_up_buf[layer]
            continue

        # PLE layers
        if suffix == "ple_conv1d.weight":
            yield f"{base}.ple.conv1d.weight", _to_bf16(t, device=device).view(-1, 1, t.shape[-1])
            continue
        if suffix == "ple_norm_conv.weight":
            yield f"{base}.ple.norm_conv.weight", _plus_one_norm(t, device=device)
            continue
        if suffix == "ple_norm_key.weight":
            yield f"{base}.ple.norm_key.weight", _plus_one_norm(t, device=device)
            continue
        if suffix == "ple_norm_query.weight":
            yield f"{base}.ple.norm_query.weight", _plus_one_norm(t, device=device)
            continue
        if suffix == "ple_key.weight":
            yield f"{base}.ple.key_proj.qweight", t.packed()
            continue
        if suffix == "ple_value.weight":
            yield f"{base}.ple.value_proj.qweight", t.packed()
            continue

        # GDN layers
        if suffix == "ssm_conv1d.weight":
            w = _to_bf16(t, device=device)
            if untile:
                # channels are [q | k | v]; only the V block is tiled
                w = torch.cat([w[:qk_rows], _ungroup_v(w[qk_rows:], 0, vK, vR, vD)], dim=0)
            yield f"{base}.linear_attn.conv1d.weight", w.view(-1, 1, t.shape[-1])
            continue
        if suffix == "ssm_norm.weight":
            yield f"{base}.linear_attn.norm.weight", _to_bf16(t)
            continue
        if suffix == "ssm_out.weight":
            if untile and _straddles_blocks(t.ggml_type, vD):
                yield f"{base}.linear_attn.out_proj.qweight", t.packed()
                perm = _tiled_input_perm(vK, vR, vD).to(device)
                yield f"{base}.linear_attn.out_proj_in_perm", perm
                continue
            w = _ungroup_packed_cols(t, vK, vR, vD) if untile else t.packed()
            yield f"{base}.linear_attn.out_proj.qweight", w
            continue
        if suffix == "ssm_a":
            a = _to_f32(t)
            if untile:
                a = _ungroup_v(a, 0, vK, vR, 1)
            yield f"{base}.linear_attn.A_log", torch.log(-a) if bool((a < 0).all()) else a
            continue
        if suffix == "ssm_dt.bias":
            dt = _to_f32(t)
            yield f"{base}.linear_attn.dt_bias", _ungroup_v(dt, 0, vK, vR, 1) if untile else dt
            continue
        if suffix in ("attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight"):
            key = suffix.split(".")[0].replace("attn_", "").replace("ssm_", "")
            w = t.packed()
            if untile:
                # whole rows permute safely on packed data (each row is its own block run)
                if key == "qkv":
                    w = torch.cat([w[:qk_rows], _ungroup_packed_rows(w[qk_rows:], vK, vR, vD)])
                else:
                    w = _ungroup_packed_rows(w, vK, vR, vD if key == "gate" else 1)
            in_proj_buf.setdefault(layer, {})[key] = w
            slots = in_proj_buf[layer]
            if len(slots) == 4:
                types = [
                    quant_map.get((layer, "attn_qkv.weight")),
                    quant_map.get((layer, "attn_gate.weight")),
                    quant_map.get((layer, "ssm_beta.weight")),
                    quant_map.get((layer, "ssm_alpha.weight")),
                ]
                if len(set(types)) == 1:
                    yield (
                        f"{base}.linear_attn.in_proj.qweight",
                        torch.cat(
                            [slots["qkv"], slots["gate"], slots["beta"], slots["alpha"]], dim=0
                        ),
                    )
                else:
                    yield f"{base}.linear_attn.in_proj.qweight_0", slots["qkv"]
                    yield f"{base}.linear_attn.in_proj.qweight_1", slots["gate"]
                    yield f"{base}.linear_attn.in_proj.qweight_2", slots["beta"]
                    yield f"{base}.linear_attn.in_proj.qweight_3", slots["alpha"]
                del in_proj_buf[layer]
            continue

        # QSA layers
        if suffix == "attn_output.weight":
            yield f"{base}.self_attn.o_proj.qweight", t.packed()
            continue
        if suffix == "attn_q_norm.weight":
            yield f"{base}.self_attn.q_norm.weight", _plus_one_norm(t)
            continue
        if suffix == "attn_k_norm.weight":
            yield f"{base}.self_attn.k_norm.weight", _plus_one_norm(t)
            continue
        if suffix in ("attn_q.weight", "attn_k.weight", "attn_v.weight"):
            k = suffix.split(".")[0].replace("attn_", "")
            qkv_buf.setdefault(layer, {})[k] = t.packed()
            if len(qkv_buf[layer]) == 3:
                slots = qkv_buf[layer]
                types = [
                    quant_map.get((layer, "attn_q.weight")),
                    quant_map.get((layer, "attn_k.weight")),
                    quant_map.get((layer, "attn_v.weight")),
                ]
                if len(set(types)) == 1:
                    yield (
                        f"{base}.self_attn.qkv_proj.qweight",
                        torch.cat([slots["q"], slots["k"], slots["v"]], dim=0),
                    )
                else:
                    yield f"{base}.self_attn.qkv_proj.qweight_0", slots["q"]
                    yield f"{base}.self_attn.qkv_proj.qweight_1", slots["k"]
                    yield f"{base}.self_attn.qkv_proj.qweight_2", slots["v"]
                del qkv_buf[layer]
            continue

        # QSA indexer
        if suffix == "indexer.q_norm.weight":
            yield f"{base}.self_attn.indexer.q_layernorm.weight", _plus_one_norm(t)
            continue
        if suffix == "indexer.k_norm.weight":
            yield f"{base}.self_attn.indexer.k_layernorm.weight", _plus_one_norm(t)
            continue
        if suffix in ("indexer.q_proj.weight", "indexer.k_proj.weight"):
            k = suffix.split(".")[1].replace("_proj", "")
            index_buf.setdefault(layer, {})[k] = t.packed()
            if len(index_buf[layer]) == 2:
                slots = index_buf[layer]
                types = [
                    quant_map.get((layer, "indexer.q_proj.weight")),
                    quant_map.get((layer, "indexer.k_proj.weight")),
                ]
                if len(set(types)) == 1:
                    yield (
                        f"{base}.self_attn.indexer.index_qk_proj.qweight",
                        torch.cat([slots["q"], slots["k"]], dim=0),
                    )
                else:
                    yield f"{base}.self_attn.indexer.index_qk_proj.qweight_0", slots["q"]
                    yield f"{base}.self_attn.indexer.index_qk_proj.qweight_1", slots["k"]
                del index_buf[layer]
            continue

    # Yield PLE embedding weights (computed, not stored in GGUF)
    if qwen4_args is not None:
        from .ple import derive_ngram_hash_constants

        for ple_index, ple_layer_id in enumerate(qwen4_args.ple_layer_ids):
            mult, sizes, offsets = derive_ngram_hash_constants(
                vocab_size=config.vocab_size,
                ngram_size=qwen4_args.ngram_size,
                num_ngram_heads=qwen4_args.num_ngram_heads,
                ngram_vocab_size_base=qwen4_args.ngram_vocab_size_base,
                ple_layer_index=ple_index,  # position among PLE layers, matching Qwen4ExpPLE.ple_index
            )
            yield (
                f"model.layers.{ple_layer_id}.ple.ple_embedding.layer_multipliers",
                torch.tensor(mult, dtype=torch.int64),
            )
            yield (
                f"model.layers.{ple_layer_id}.ple.ple_embedding.ngram_heads_vocab_sizes",
                torch.tensor(sizes, dtype=torch.int64),
            )
            yield (
                f"model.layers.{ple_layer_id}.ple.ple_embedding.ngram_heads_offsets",
                torch.tensor(offsets, dtype=torch.int64),
            )

    # Vision tower: streamed from the external mmproj GGUF file, named exactly as
    # iter_vision_weights names the HF checkpoint's visual.* tensors.
    if include_vision and config.vision_config is not None:
        from .mmproj import discover_mmproj_path, iter_mmproj_vision_weights

        mmproj_path = discover_mmproj_path(model_path)
        if mmproj_path is not None:
            yield from iter_mmproj_vision_weights(mmproj_path, device)


def load_ple_table_from_gguf(
    model_path: str,
    qwen4_args,
    *,
    pin: bool = False,
):
    """Map the GGUF per_layer_token_embd.weight table directly via mmap without blowing host RAM."""
    from freetoken.models.gguf.reader import find_gguf_tensor
    from .ple import GgufUVATable

    found = find_gguf_tensor(model_path, "per_layer_token_embd.weight")
    if found is None:
        raise ValueError(f"per_layer_token_embd.weight not found in GGUF files for {model_path}")
    t = found[1]
    return GgufUVATable(
        torch.from_numpy(t.data),
        quant_type=int(t.tensor_type),
        embed_dim=qwen4_args.ngram_head_dim,
    )


def iter_gguf_mtp_weights(
    model_path: str,
    device: torch.device,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield MTP weights from GGUF for qwen4_exp."""
    mtp_path = _find_mtp_gguf_path(model_path)
    if mtp_path is None:
        return

    from freetoken.models.gguf.reader import iter_gguf_tensors

    quant_map = _scan_quant_types(mtp_path)
    qkv_buf: dict[int, dict[str, torch.Tensor]] = {}
    index_buf: dict[int, dict[str, torch.Tensor]] = {}
    gate_up_buf: dict[int, dict[str, torch.Tensor]] = {}
    hc_attn_down_buf: dict[int, dict[str, torch.Tensor]] = {}
    hc_ffn_down_buf: dict[int, dict[str, torch.Tensor]] = {}

    for t in iter_gguf_tensors(mtp_path):
        name = t.name
        if (
            ".mlp.experts." in name
            or "ffn_gate_exps" in name
            or "ffn_down_exps" in name
            or "ffn_up_exps" in name
        ):
            continue

        if not name.startswith("blk."):
            continue

        parts = name.split(".", 2)
        layer = int(parts[1])
        suffix = parts[2]
        base = "mtp.layers.0"

        # Handle nextn prefix tensors
        if suffix.startswith("nextn."):
            nextn_suffix = suffix[len("nextn.") :]
            if nextn_suffix == "eh_proj.weight":
                raw = _to_bf16(t, device=device)
                hidden_dim = raw.shape[0]
                yield "mtp.fc_embedding.weight", raw[:, :hidden_dim]
                # A split export stores the hidden half as its own nextn.fc_hidden tensor.
                if raw.shape[1] > hidden_dim:
                    yield "mtp.fc_hidden.weight", raw[:, hidden_dim:]
                continue
            if nextn_suffix == "fc_hidden.weight":
                yield "mtp.fc_hidden.weight", _to_bf16(t, device=device)
                continue
            if nextn_suffix == "enorm.weight":
                yield "mtp.pre_fc_norm_embedding.weight", _plus_one_norm(t, device=device)
                continue
            if nextn_suffix == "hnorm.weight":
                yield "mtp.pre_fc_norm_hidden.weight", _plus_one_norm(t, device=device)
                continue
            if nextn_suffix == "hc_head_norm.weight":
                yield "mtp.hyper_connection_mixer.hc_norm.weight", _plus_one_norm(t, device=device)
                continue
            if nextn_suffix == "hc_head_up.weight":
                yield (
                    "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
                    _to_bf16(t, device=device),
                )
                continue
            if nextn_suffix == "hc_head_down.weight":
                yield (
                    "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
                    _to_bf16(t, device=device),
                )
                continue

        # HC norm & up projections
        if suffix == "hc_attn_norm.weight":
            yield f"{base}.attn_hyper_connection.hc_norm.weight", _plus_one_norm(t, device=device)
            continue
        if suffix == "hc_ffn_norm.weight":
            yield f"{base}.mlp_hyper_connection.hc_norm.weight", _plus_one_norm(t, device=device)
            continue
        if suffix == "hc_attn_up.weight":
            yield (
                f"{base}.attn_hyper_connection.input_mix_weight_up.weight",
                _to_bf16(t, device=device),
            )
            continue
        if suffix == "hc_ffn_up.weight":
            yield (
                f"{base}.mlp_hyper_connection.input_mix_weight_up.weight",
                _to_bf16(t, device=device),
            )
            continue

        # HC down + inject fusion
        if suffix == "hc_attn_down.weight":
            hc_attn_down_buf.setdefault(layer, {})["down"] = _to_bf16(t, device=device)
        elif suffix == "hc_attn_inject.weight":
            hc_attn_down_buf.setdefault(layer, {})["inject"] = _to_bf16(t, device=device)
        if (
            layer in hc_attn_down_buf
            and "down" in hc_attn_down_buf[layer]
            and "inject" in hc_attn_down_buf[layer]
        ):
            d = hc_attn_down_buf[layer]["down"].to(device)
            inj = hc_attn_down_buf[layer]["inject"].to(device)
            pad = torch.zeros(12, d.shape[1], dtype=d.dtype, device=device)
            yield (
                f"{base}.attn_hyper_connection.input_mix_weight_down_block_inject.weight",
                torch.cat([d, inj, pad], dim=0),
            )
            del hc_attn_down_buf[layer]
            continue

        if suffix == "hc_ffn_down.weight":
            hc_ffn_down_buf.setdefault(layer, {})["down"] = _to_bf16(t, device=device)
        elif suffix == "hc_ffn_inject.weight":
            hc_ffn_down_buf.setdefault(layer, {})["inject"] = _to_bf16(t, device=device)
        if (
            layer in hc_ffn_down_buf
            and "down" in hc_ffn_down_buf[layer]
            and "inject" in hc_ffn_down_buf[layer]
        ):
            d = hc_ffn_down_buf[layer]["down"].to(device)
            inj = hc_ffn_down_buf[layer]["inject"].to(device)
            pad = torch.zeros(12, d.shape[1], dtype=d.dtype, device=device)
            yield (
                f"{base}.mlp_hyper_connection.input_mix_weight_down_block_inject.weight",
                torch.cat([d, inj, pad], dim=0),
            )
            del hc_ffn_down_buf[layer]
            continue

        # MoE router & shared expert
        if suffix == "ffn_gate_inp.weight":
            yield f"{base}.mlp.gate.weight", _to_bf16(t, device=device)
            continue
        if suffix == "ffn_gate_inp_shexp.weight":
            yield f"{base}.mlp.shared_expert_gate.weight", _to_bf16(t, device=device).reshape(1, -1)
            continue
        if suffix == "ffn_down_shexp.weight":
            yield f"{base}.mlp.shared_expert.down_proj.weight", _to_bf16(t, device=device)
            continue
        if suffix == "ffn_gate_shexp.weight":
            gate_up_buf.setdefault(layer, {})["gate"] = _to_bf16(t, device=device)
        elif suffix == "ffn_up_shexp.weight":
            gate_up_buf.setdefault(layer, {})["up"] = _to_bf16(t, device=device)
        if layer in gate_up_buf and "gate" in gate_up_buf[layer] and "up" in gate_up_buf[layer]:
            gu = gate_up_buf[layer]
            yield (
                f"{base}.mlp.shared_expert.gate_up_proj.weight",
                torch.cat([gu["gate"], gu["up"]], dim=0),
            )
            del gate_up_buf[layer]
            continue

        # QSA layers
        if suffix == "attn_output.weight":
            yield f"{base}.self_attn.o_proj.weight", _to_bf16(t, device=device)
            continue
        if suffix == "attn_q_norm.weight":
            yield f"{base}.self_attn.q_norm.weight", _plus_one_norm(t, device=device)
            continue
        if suffix == "attn_k_norm.weight":
            yield f"{base}.self_attn.k_norm.weight", _plus_one_norm(t, device=device)
            continue
        if suffix in ("attn_q.weight", "attn_k.weight", "attn_v.weight"):
            k = suffix.split(".")[0].replace("attn_", "")
            qkv_buf.setdefault(layer, {})[k] = _to_bf16(t, device=device)
            if len(qkv_buf[layer]) == 3:
                slots = qkv_buf[layer]
                yield (
                    f"{base}.self_attn.qkv_proj.weight",
                    torch.cat([slots["q"], slots["k"], slots["v"]], dim=0),
                )
                del qkv_buf[layer]
            continue

        # QSA indexer
        if suffix == "indexer.q_norm.weight":
            yield f"{base}.self_attn.indexer.q_layernorm.weight", _plus_one_norm(t, device=device)
            continue
        if suffix == "indexer.k_norm.weight":
            yield f"{base}.self_attn.indexer.k_layernorm.weight", _plus_one_norm(t, device=device)
            continue
        if suffix in ("indexer.q_proj.weight", "indexer.k_proj.weight"):
            k = suffix.split(".")[1].replace("_proj", "")
            index_buf.setdefault(layer, {})[k] = _to_bf16(t, device=device)
            if len(index_buf[layer]) == 2:
                slots = index_buf[layer]
                yield (
                    f"{base}.self_attn.indexer.index_qk_proj.weight",
                    torch.cat([slots["q"], slots["k"]], dim=0),
                )
                del index_buf[layer]
            continue
