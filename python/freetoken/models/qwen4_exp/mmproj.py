"""External llama.cpp mmproj GGUF: vision tower for the GGUF qwen4exp checkpoint.

Qwen3.8-Flash-Next's GGUF release ships the vision tower as a separate ``mmproj-*.gguf``
file (arch ``clip``, projector type ``qwen3vl_merger``) instead of folding it into the
main checkpoint like the HF (NVFP4) release does. This module discovers that file,
synthesizes the same :class:`VisionConfig` the HF loader builds from
``config.json``'s ``vision_config``, and streams its tensors under the same
``visual.*`` names :func:`freetoken.models.qwen4_exp.weight.iter_vision_weights` uses.
"""

from __future__ import annotations

import glob
import os
from typing import Iterator

import torch

from freetoken.models.qwen3_vl.config import VisionConfig

from .gguf import _dense

# env var bridge for the ``--mmproj`` CLI override: GgufConfigShim / parse_gguf_config get
# only (model_path) / (shim), so a value from ServerArgs travels through this instead of
# widening the shared parse_config(hf_config) signature every family uses.
MMPROJ_PATH_ENV = "FREETOKEN_MMPROJ_PATH"

_PROJECTOR_TYPE = "qwen3vl_merger"
# BF16 > F16 > any other mmproj file, matched case-insensitively against the basename.
_PREFERENCE = ("bf16", "f16")


def _mmproj_candidates(model_dir: str) -> list[str]:
    seen: dict[str, None] = {}
    for d in (model_dir, os.path.dirname(os.path.normpath(model_dir))):
        for path in sorted(glob.glob(os.path.join(d, "mmproj*.gguf"))):
            seen.setdefault(path, None)
    return list(seen)


def _rank(path: str) -> int:
    base = os.path.basename(path).lower()
    for i, tag in enumerate(_PREFERENCE):
        if tag in base:
            return i
    return len(_PREFERENCE)


def discover_mmproj_path(model_path: str, *, override: str | None = None) -> str | None:
    """The mmproj GGUF file to use, or ``None`` when none is found/configured.

    ``override`` (``--mmproj`` / ``FREETOKEN_MMPROJ_PATH``) is validated and returned as-is.
    Otherwise, candidates are ``mmproj*.gguf`` in the model file's own directory and its
    parent, filtered to GGUF arch ``"clip"``, preferring BF16 over F16 over any other file.
    """
    from freetoken.models.gguf.reader import gguf_architecture, resolve_gguf_path

    if override is None:
        override = os.environ.get(MMPROJ_PATH_ENV) or None
    if override is not None:
        if gguf_architecture(override) != "clip":
            raise ValueError(f"--mmproj {override!r} is not a clip (mmproj) GGUF file")
        return override

    main_path = resolve_gguf_path(model_path)
    if main_path is None:
        return None
    model_dir = os.path.dirname(main_path)
    candidates = []
    for path in _mmproj_candidates(model_dir):
        try:
            if gguf_architecture(path) == "clip":
                candidates.append(path)
        except Exception:  # noqa: BLE001 -- an unrelated/corrupt sibling must not abort discovery
            continue
    if not candidates:
        return None
    return min(candidates, key=_rank)


def _kv(metadata: dict, key: str, default=None):
    if key in metadata:
        return metadata[key]
    if default is not None:
        return default
    raise KeyError(f"mmproj GGUF metadata missing key {key!r}")


def read_mmproj_vision_config(mmproj_path: str) -> VisionConfig:
    """Synthesize the HF-equivalent ``VisionConfig`` from the mmproj GGUF's ``clip.*`` metadata."""
    from freetoken.models.gguf.reader import load_gguf_metadata

    md = load_gguf_metadata(mmproj_path)
    projector_type = _kv(md, "clip.projector_type")
    if projector_type != _PROJECTOR_TYPE:
        raise ValueError(
            f"mmproj {mmproj_path!r}: projector_type {projector_type!r} is not supported "
            f"(only {_PROJECTOR_TYPE!r})"
        )
    deepstack = _kv(md, "clip.vision.is_deepstack_layers", [])
    if any(deepstack):
        raise ValueError(f"mmproj {mmproj_path!r}: DeepStack layers are not supported")
    patch_size = int(_kv(md, "clip.vision.patch_size"))
    image_size = int(_kv(md, "clip.vision.image_size"))
    return VisionConfig(
        hidden_size=int(_kv(md, "clip.vision.embedding_length")),
        depth=int(_kv(md, "clip.vision.block_count")),
        num_heads=int(_kv(md, "clip.vision.attention.head_count")),
        intermediate_size=int(_kv(md, "clip.vision.feed_forward_length")),
        patch_size=patch_size,
        temporal_patch_size=2,
        spatial_merge_size=int(_kv(md, "clip.vision.spatial_merge_size")),
        num_position_embeddings=(image_size // patch_size) ** 2,
        out_hidden_size=int(_kv(md, "clip.vision.projection_dim")),
        in_channels=3,
        deepstack_visual_indexes=(),
    )


# GGUF ``v.blk.N.<suffix>`` -> FreeToken ``visual.blocks.N.<suffix>`` block tensor names.
_BLOCK_SUFFIX_MAP = {
    "ln1": "norm1",
    "ln2": "norm2",
    "attn_qkv": "attn.qkv",
    "attn_out": "attn.proj",
    "ffn_up": "mlp.linear_fc1",
    "ffn_down": "mlp.linear_fc2",
}
# Non-block GGUF ``weight``/``bias`` pairs -> FreeToken ``visual.<name>`` base name.
_TOP_LEVEL_PAIR_MAP = {
    "v.post_ln": "merger.norm",
    "mm.0": "merger.linear_fc1",
    "mm.2": "merger.linear_fc2",
}
# Single (unpaired) GGUF tensor name -> full FreeToken name.
_TOP_LEVEL_SINGLE_MAP = {
    "v.position_embd.weight": "visual.pos_embed.weight",
    "v.patch_embd.bias": "visual.patch_embed.proj.bias",
}


def iter_mmproj_vision_weights(
    mmproj_path: str, device: torch.device
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the vision tower's weights from the mmproj GGUF, named exactly as
    :func:`freetoken.models.qwen4_exp.weight.iter_vision_weights` names the HF checkpoint's
    ``visual.*`` tensors, so both sources feed the same state-dict load path."""
    from freetoken.models.gguf.reader import iter_gguf_tensors

    tensors = {t.name: t for t in iter_gguf_tensors(mmproj_path)}

    w0 = tensors.pop("v.patch_embd.weight", None)
    w1 = tensors.pop("v.patch_embd.weight.1", None)
    if w0 is not None and w1 is not None:
        stacked = torch.stack(
            [_dense(w0, torch.bfloat16, device=device), _dense(w1, torch.bfloat16, device=device)],
            dim=2,
        )
        yield "visual.patch_embed.proj.weight", stacked

    for name, t in tensors.items():
        if name.startswith("v.blk."):
            _, _, layer, suffix = name.split(".", 3)
            base_suffix, kind = suffix.rsplit(".", 1)  # "attn_qkv", "weight" | "bias"
            target = _BLOCK_SUFFIX_MAP.get(base_suffix)
            if target is None:
                continue
            yield f"visual.blocks.{layer}.{target}.{kind}", _dense(t, torch.bfloat16, device=device)
            continue
        if name in _TOP_LEVEL_SINGLE_MAP:
            yield _TOP_LEVEL_SINGLE_MAP[name], _dense(t, torch.bfloat16, device=device)
            continue
        base, _, kind = name.rpartition(".")
        if base in _TOP_LEVEL_PAIR_MAP:
            yield (
                f"visual.{_TOP_LEVEL_PAIR_MAP[base]}.{kind}",
                _dense(t, torch.bfloat16, device=device),
            )


__all__ = [
    "MMPROJ_PATH_ENV",
    "discover_mmproj_path",
    "read_mmproj_vision_config",
    "iter_mmproj_vision_weights",
]


# Qwen3-VL pixel budget for the qwen3vl_merger projector (HF Qwen3VLProcessor defaults); the
# clip metadata carries patch/merge/normalization but not the resize bounds.
_QWEN3VL_MIN_PIXELS = 65536
_QWEN3VL_MAX_PIXELS = 16777216


def mmproj_image_processor(mmproj_path: str):
    """A Qwen2VL-family image processor configured from the mmproj clip metadata."""
    from transformers import Qwen2VLImageProcessor

    from freetoken.models.gguf.reader import load_gguf_metadata

    meta = load_gguf_metadata(mmproj_path)
    return Qwen2VLImageProcessor(
        size={"shortest_edge": _QWEN3VL_MIN_PIXELS, "longest_edge": _QWEN3VL_MAX_PIXELS},
        patch_size=int(meta["clip.vision.patch_size"]),
        temporal_patch_size=2,
        merge_size=int(meta["clip.vision.spatial_merge_size"]),
        image_mean=_triple(meta["clip.vision.image_mean"]),
        image_std=_triple(meta["clip.vision.image_std"]),
    )


def _triple(value) -> list[float]:
    values = [float(x) for x in (value if isinstance(value, (list, tuple)) else [value])]
    return values * 3 if len(values) == 1 else values
