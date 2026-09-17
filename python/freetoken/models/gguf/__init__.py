from .dequant import GGML_NAME, dequantize
from .reader import (
    FTW_METADATA_GGUF,
    GgufTensor,
    gguf_architecture,
    gguf_config_source,
    gguf_tensor_names,
    is_gguf_path,
    iter_gguf_tensors,
    load_gguf_metadata,
    write_metadata_gguf,
)

# (model_path, reason) -> warned; a loader consumes every tensor once per load.
_warned_dropped: set[tuple[str, str]] = set()


def warn_dropped_tensors(model_path: str, reason: str, detail: str) -> None:
    """Name GGUF tensors the loader is dropping, once per checkpoint and reason.

    A GGUF file can carry weights the served model has no place for (a NextN/MTP draft
    block, which this path never speculates with). Dropping one silently would let a
    checkpoint that is not what the user thinks it is load anyway, so every drop site
    reports it here instead of just skipping.
    """
    from freetoken.utils import init_logger

    if (model_path, reason) in _warned_dropped:
        return
    _warned_dropped.add((model_path, reason))
    init_logger(__name__).warning_rank0(f"GGUF {model_path}: {detail}")


__all__ = [
    "GGML_NAME",
    "dequantize",
    "FTW_METADATA_GGUF",
    "GgufTensor",
    "gguf_architecture",
    "gguf_config_source",
    "gguf_tensor_names",
    "is_gguf_path",
    "iter_gguf_tensors",
    "load_gguf_metadata",
    "warn_dropped_tensors",
    "write_metadata_gguf",
]
