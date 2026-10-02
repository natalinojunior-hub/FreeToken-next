from .config import parse_config
from .model import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)
from .weight import (
    iter_expert_pieces,
    iter_vision_weights,
    iter_weights,
    iter_weights_parallel,
    nvfp4_expert_spec,
)
from .gguf import iter_gguf_mtp_weights, parse_gguf_config, iter_gguf_weights
from .mtp import MTP_PATH_ENV, has_hf_mtp_weights, is_hf_mtp_head, iter_hf_mtp_weights


def iter_mtp_weights(model_path: str, device, *, experts: bool = True):
    """Engine hook for an external safetensors head or in-file GGUF NextN block."""
    import os

    external = os.environ.get(MTP_PATH_ENV)
    if external and external.endswith(".safetensors"):
        return iter_hf_mtp_weights(external, device)
    if has_hf_mtp_weights(model_path):
        return iter_hf_mtp_weights(model_path, device, experts=experts)
    return iter_gguf_mtp_weights(model_path, device)


# Resolved off this package by freetoken.moe.expert_banks._gguf_banks (the GGUF expert
# layout is architecture-specific, so the provider looks it up via the model registry).
from .gguf_experts import gguf_expert_types, load_gguf_expert_sources

__all__ = [
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "parse_config",
    "iter_vision_weights",
    "iter_weights",
    "iter_weights_parallel",
    "iter_expert_pieces",
    "nvfp4_expert_spec",
    "parse_gguf_config",
    "iter_gguf_weights",
    "iter_gguf_mtp_weights",
    "iter_mtp_weights",
    "is_hf_mtp_head",
    "has_hf_mtp_weights",
    "gguf_expert_types",
    "load_gguf_expert_sources",
]
