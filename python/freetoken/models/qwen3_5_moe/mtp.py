"""Standalone Qwen3.5-MoE HF MTP sidecar reader."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterator

import safetensors
import torch

MTP_PATH_ENV = "FREETOKEN_MTP_PATH"

_REQUIRED = {
    "mtp.fc.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.norm.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp.experts.gate_up_proj",
    "mtp.layers.0.mlp.experts.down_proj",
}
_GEMMA_NORMS = {
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.norm.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
}
_EXPERT_KEY = re.compile(
    r"^mtp\.layers\.0\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


def has_hf_mtp_weights(path: str | None) -> bool:
    """Recognize one-layer HF MTP weights in a sidecar or model checkpoint."""
    if is_hf_mtp_head(path):
        return True
    if not path or not os.path.isdir(path):
        return False
    index = Path(path) / "model.safetensors.index.json"
    try:
        if index.is_file():
            import json

            keys = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
        else:
            keys = set()
            for file in Path(path).glob("*.safetensors"):
                with safetensors.safe_open(str(file), framework="pt", device="cpu") as f:
                    keys.update(f.keys())
        if not _HEAD_REQUIRED <= keys:
            return False
        if {
            "mtp.layers.0.mlp.experts.gate_up_proj",
            "mtp.layers.0.mlp.experts.down_proj",
        } <= keys:
            return True
        expert_parts = {
            (int(match[1]), match[2].removesuffix("_proj"))
            for key in keys
            if (match := _EXPERT_KEY.match(key))
        }
        expert_ids = {expert for expert, _ in expert_parts}
        return bool(expert_ids) and expert_parts == {
            (expert, part) for expert in range(len(expert_ids)) for part in ("gate", "up", "down")
        }
    except (OSError, KeyError, ValueError, safetensors.SafetensorError):
        return False


_HEAD_REQUIRED = _REQUIRED - {
    "mtp.layers.0.mlp.experts.gate_up_proj",
    "mtp.layers.0.mlp.experts.down_proj",
}


def is_hf_mtp_head(path: str | None) -> bool:
    """Recognize this family's standalone safetensors head by its parameter schema."""
    if not path or not os.path.isfile(path) or not path.endswith(".safetensors"):
        return False
    try:
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            return _REQUIRED <= set(f.keys())
    except (OSError, safetensors.SafetensorError):
        return False


def iter_hf_mtp_weights(path: str, device: torch.device) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield sidecar or in-checkpoint MTP tensors in the model's state-dict layout."""
    standalone = is_hf_mtp_head(path)
    if not standalone and not has_hf_mtp_weights(path):
        raise ValueError(f"not a supported standalone Qwen3.5-MoE MTP safetensors head: {path}")

    direct = {
        "mtp.fc.weight": "mtp.eh_proj.weight",
        "mtp.pre_fc_norm_embedding.weight": "mtp.enorm.weight",
        "mtp.pre_fc_norm_hidden.weight": "mtp.hnorm.weight",
        "mtp.norm.weight": "mtp.shared_head_norm.weight",
        "mtp.layers.0.input_layernorm.weight": "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight": "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.layers.0.self_attn.o_proj.weight": "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.self_attn.q_norm.weight": "mtp.layers.0.self_attn.q_norm.weight",
        "mtp.layers.0.self_attn.k_norm.weight": "mtp.layers.0.self_attn.k_norm.weight",
        "mtp.layers.0.mlp.gate.weight": "mtp.layers.0.mlp.gate.weight",
        "mtp.layers.0.mlp.shared_expert_gate.weight": "mtp.layers.0.mlp.shared_expert_gate.weight",
        "mtp.layers.0.mlp.shared_expert.down_proj.weight": "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        "mtp.layers.0.mlp.experts.gate_up_proj": "mtp.layers.0.mlp.experts.gate_up_proj",
        "mtp.layers.0.mlp.experts.down_proj": "mtp.layers.0.mlp.experts.down_proj",
    }
    from freetoken.models.loader import iter_weight_files

    source = [path] if standalone else iter_weight_files(path)
    required = (
        set(direct)
        - {
            "mtp.layers.0.mlp.experts.gate_up_proj",
            "mtp.layers.0.mlp.experts.down_proj",
        }
        | {f"mtp.layers.0.self_attn.{part}_proj.weight" for part in ("q", "k", "v")}
        | {
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
            "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        }
    )
    values: dict[str, torch.Tensor] = {}
    expert_keys: dict[tuple[int, str], torch.Tensor] = {}
    for file in source:
        with safetensors.safe_open(str(file), framework="pt", device=str(device)) as f:
            for key in f.keys():
                if key in required or key in {
                    "mtp.layers.0.mlp.experts.gate_up_proj",
                    "mtp.layers.0.mlp.experts.down_proj",
                }:
                    values[key] = f.get_tensor(key)
                else:
                    match = _EXPERT_KEY.match(key)
                    if match:
                        expert_keys[(int(match[1]), match[2].removesuffix("_proj"))] = f.get_tensor(
                            key
                        )
    missing = required - values.keys()
    if missing:
        raise ValueError(f"incomplete Qwen3.5-MoE MTP weights: {sorted(missing)}")
    for key, target in direct.items():
        if key not in values:
            continue
        tensor = values[key]
        if key in _GEMMA_NORMS:
            tensor = tensor + 1.0
        yield target, tensor
    qkv = [values[f"mtp.layers.0.self_attn.{part}_proj.weight"] for part in ("q", "k", "v")]
    yield "mtp.layers.0.self_attn.qkv_proj.weight", torch.cat(qkv, dim=0)
    gate_up = torch.cat(
        [
            values["mtp.layers.0.mlp.shared_expert.gate_proj.weight"],
            values["mtp.layers.0.mlp.shared_expert.up_proj.weight"],
        ],
        dim=0,
    )
    yield "mtp.layers.0.mlp.shared_expert.gate_up_proj.weight", gate_up
    if "mtp.layers.0.mlp.experts.gate_up_proj" in values:
        yield (
            "mtp.layers.0.mlp.experts.gate_up_proj",
            values["mtp.layers.0.mlp.experts.gate_up_proj"],
        )
        yield "mtp.layers.0.mlp.experts.down_proj", values["mtp.layers.0.mlp.experts.down_proj"]
    elif expert_keys:
        ids = sorted({expert for expert, _ in expert_keys})
        yield (
            "mtp.layers.0.mlp.experts.gate_up_proj",
            torch.stack(
                [torch.cat([expert_keys[(i, "gate")], expert_keys[(i, "up")]], dim=0) for i in ids]
            ),
        )
        yield (
            "mtp.layers.0.mlp.experts.down_proj",
            torch.stack([expert_keys[(i, "down")] for i in ids]),
        )


__all__ = ["MTP_PATH_ENV", "has_hf_mtp_weights", "is_hf_mtp_head", "iter_hf_mtp_weights"]
