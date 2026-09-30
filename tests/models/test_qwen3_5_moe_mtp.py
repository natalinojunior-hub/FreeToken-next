from __future__ import annotations

import torch
from safetensors.torch import save_file

import json

from freetoken.models.qwen3_5_moe.mtp import (
    has_hf_mtp_weights,
    is_hf_mtp_head,
    iter_hf_mtp_weights,
)
from freetoken.engine.config import _safe_spec_mtp_depth


def test_qwen35_mtp_is_fail_closed_until_state_parity():
    from types import SimpleNamespace

    assert _safe_spec_mtp_depth(SimpleNamespace(model_type="qwen3_5_moe"), 6) == 0
    assert _safe_spec_mtp_depth(SimpleNamespace(model_type="qwen4_exp"), 6) == 6
    assert _safe_spec_mtp_depth(SimpleNamespace(model_type="qwen3_5_moe"), 0) == 0


def test_qwen4_iq2s_iq4nl_mtp_is_fail_closed_until_token_parity():
    from types import SimpleNamespace

    assert _safe_spec_mtp_depth(
        SimpleNamespace(model_type="qwen4_exp", gguf_expert_types=(22, 20)), 5
    ) == 0


def test_standalone_mtp_sidecar_maps_to_model_state(tmp_path):
    path = tmp_path / "model_mtp.safetensors"
    source = {
        "mtp.fc.weight": torch.zeros(4, 8),
        "mtp.pre_fc_norm_embedding.weight": torch.zeros(4),
        "mtp.pre_fc_norm_hidden.weight": torch.zeros(4),
        "mtp.norm.weight": torch.zeros(4),
        "mtp.layers.0.input_layernorm.weight": torch.zeros(4),
        "mtp.layers.0.post_attention_layernorm.weight": torch.zeros(4),
        "mtp.layers.0.self_attn.q_proj.weight": torch.zeros(4, 4),
        "mtp.layers.0.self_attn.k_proj.weight": torch.zeros(2, 4),
        "mtp.layers.0.self_attn.v_proj.weight": torch.zeros(2, 4),
        "mtp.layers.0.self_attn.o_proj.weight": torch.zeros(4, 4),
        "mtp.layers.0.self_attn.q_norm.weight": torch.zeros(2),
        "mtp.layers.0.self_attn.k_norm.weight": torch.zeros(2),
        "mtp.layers.0.mlp.gate.weight": torch.zeros(3, 4),
        "mtp.layers.0.mlp.shared_expert_gate.weight": torch.zeros(1, 4),
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": torch.zeros(2, 4),
        "mtp.layers.0.mlp.shared_expert.up_proj.weight": torch.zeros(2, 4),
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": torch.zeros(4, 2),
        "mtp.layers.0.mlp.experts.gate_up_proj": torch.zeros(3, 4, 4),
        "mtp.layers.0.mlp.experts.down_proj": torch.zeros(3, 4, 2),
    }
    save_file(source, str(path))

    assert is_hf_mtp_head(str(path))
    mapped = dict(iter_hf_mtp_weights(str(path), torch.device("cpu")))
    assert mapped["mtp.layers.0.self_attn.qkv_proj.weight"].shape == (8, 4)
    assert mapped["mtp.layers.0.mlp.shared_expert.gate_up_proj.weight"].shape == (4, 4)
    assert torch.equal(mapped["mtp.enorm.weight"], torch.ones(4))
    assert torch.equal(mapped["mtp.layers.0.self_attn.q_norm.weight"], torch.ones(2))
    assert mapped["mtp.layers.0.mlp.experts.gate_up_proj"].shape == (3, 4, 4)


def test_integrated_mtp_head_stacks_per_expert_weights(tmp_path):
    source = {
        "mtp.fc.weight": torch.zeros(4, 8),
        "mtp.pre_fc_norm_embedding.weight": torch.zeros(4),
        "mtp.pre_fc_norm_hidden.weight": torch.zeros(4),
        "mtp.norm.weight": torch.zeros(4),
        "mtp.layers.0.input_layernorm.weight": torch.zeros(4),
        "mtp.layers.0.post_attention_layernorm.weight": torch.zeros(4),
        "mtp.layers.0.self_attn.q_proj.weight": torch.zeros(4, 4),
        "mtp.layers.0.self_attn.k_proj.weight": torch.zeros(2, 4),
        "mtp.layers.0.self_attn.v_proj.weight": torch.zeros(2, 4),
        "mtp.layers.0.self_attn.o_proj.weight": torch.zeros(4, 4),
        "mtp.layers.0.self_attn.q_norm.weight": torch.zeros(2),
        "mtp.layers.0.self_attn.k_norm.weight": torch.zeros(2),
        "mtp.layers.0.mlp.gate.weight": torch.zeros(3, 4),
        "mtp.layers.0.mlp.shared_expert_gate.weight": torch.zeros(1, 4),
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight": torch.zeros(2, 4),
        "mtp.layers.0.mlp.shared_expert.up_proj.weight": torch.zeros(2, 4),
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": torch.zeros(4, 2),
    }
    for expert in range(2):
        source[f"mtp.layers.0.mlp.experts.{expert}.gate_proj.weight"] = torch.full(
            (3, 4), expert + 1.0
        )
        source[f"mtp.layers.0.mlp.experts.{expert}.up_proj.weight"] = torch.full(
            (3, 4), expert + 2.0
        )
        source[f"mtp.layers.0.mlp.experts.{expert}.down_proj.weight"] = torch.full(
            (4, 3), expert + 3.0
        )
    shard = tmp_path / "model.safetensors"
    save_file(source, str(shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard.name for key in source}}), encoding="utf-8"
    )

    assert has_hf_mtp_weights(str(tmp_path))
    mapped = dict(iter_hf_mtp_weights(str(tmp_path), torch.device("cpu")))
    assert mapped["mtp.layers.0.mlp.experts.gate_up_proj"].shape == (2, 6, 4)
    assert mapped["mtp.layers.0.mlp.experts.down_proj"].shape == (2, 4, 3)
    assert torch.equal(
        mapped["mtp.layers.0.mlp.experts.gate_up_proj"][1, :3], torch.full((3, 4), 2.0)
    )
