"""qwen4exp GGUF vision (external llama.cpp mmproj file):

(a) config synthesis matches the native HF vision_config dict bit-for-bit
(b) GGUF tensor -> FreeToken name mapping + patch_embed temporal stack match the HF
    safetensors tensors bit-exact
(c) mmproj discovery precedence (BF16 > F16)
(d) --mmproj CLI flag parsing

(a)/(b) need the real checkpoints and are skipped when they are not present on disk.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import safetensors
import torch

from freetoken.models.qwen3_vl.config import VisionConfig
from freetoken.models.qwen4_exp.mmproj import (
    MMPROJ_PATH_ENV,
    discover_mmproj_path,
    iter_mmproj_vision_weights,
    read_mmproj_vision_config,
)
from freetoken.server.args import parse_args

GGUF_MODEL_DIR = os.environ.get(
    "FREETOKEN_QWEN4EXP_GGUF_MODEL", "/models/Qwen3.8-Flash-Next-Unsloth-IQ4_XS/UD-IQ4_XS"
)
HF_MODEL_DIR = os.environ.get(
    "FREETOKEN_QWEN4EXP_HF_MODEL", "/models/Qwen3.8-Flash-Next-NVFP4-Radix"
)
MMPROJ_BF16 = os.environ.get(
    "FREETOKEN_QWEN4EXP_MMPROJ",
    "/models/Qwen3.8-Flash-Next-Unsloth-IQ4_XS/mmproj-BF16.gguf",
)

needs_gguf = pytest.mark.skipif(
    not os.path.isdir(GGUF_MODEL_DIR), reason="FREETOKEN_QWEN4EXP_GGUF_MODEL not present"
)
needs_hf = pytest.mark.skipif(
    not os.path.isdir(HF_MODEL_DIR), reason="FREETOKEN_QWEN4EXP_HF_MODEL not present"
)
needs_mmproj = pytest.mark.skipif(
    not os.path.isfile(MMPROJ_BF16), reason="FREETOKEN_QWEN4EXP_MMPROJ not present"
)

# The native HF folder's config.json["vision_config"], reproduced here as the ground truth
# (see the task's FACTS): what read_mmproj_vision_config must reproduce from GGUF metadata.
NATIVE_VISION_CONFIG = dict(
    hidden_size=1152,
    depth=27,
    num_heads=16,
    intermediate_size=4304,
    patch_size=16,
    temporal_patch_size=2,
    spatial_merge_size=2,
    num_position_embeddings=2304,
    out_hidden_size=2560,
    in_channels=3,
    deepstack_visual_indexes=(),
)


# ---------------------------------------------------------------------------
# (a) config synthesis
# ---------------------------------------------------------------------------


@needs_mmproj
def test_vision_config_matches_native():
    vc = read_mmproj_vision_config(MMPROJ_BF16)
    assert vc == VisionConfig(**NATIVE_VISION_CONFIG)


def test_vision_config_rejects_wrong_projector_type():
    def fake_metadata(_path):
        return {
            "clip.projector_type": "mlp",
            "clip.vision.is_deepstack_layers": [False],
        }

    with patch("freetoken.models.gguf.reader.load_gguf_metadata", fake_metadata):
        with pytest.raises(ValueError, match="projector_type"):
            read_mmproj_vision_config("fake.gguf")


def test_vision_config_rejects_deepstack():
    def fake_metadata(_path):
        return {
            "clip.projector_type": "qwen3vl_merger",
            "clip.vision.is_deepstack_layers": [False, True],
        }

    with patch("freetoken.models.gguf.reader.load_gguf_metadata", fake_metadata):
        with pytest.raises(ValueError, match="DeepStack"):
            read_mmproj_vision_config("fake.gguf")


# ---------------------------------------------------------------------------
# (b) weight mapping + patch_embed temporal stack, bit-exact vs. the HF checkpoint
# ---------------------------------------------------------------------------


@needs_mmproj
@needs_hf
def test_vision_weights_match_hf_bit_exact():
    hf_file = os.path.join(HF_MODEL_DIR, "model-bf16-00001.safetensors")
    if not os.path.isfile(hf_file):
        pytest.skip(f"{hf_file} not present")

    wanted = {
        "visual.blocks.0.norm1.weight",
        "visual.blocks.0.norm1.bias",
        "visual.blocks.0.attn.qkv.weight",
        "visual.blocks.0.attn.qkv.bias",
        "visual.blocks.0.attn.proj.weight",
        "visual.blocks.0.mlp.linear_fc1.weight",
        "visual.blocks.0.mlp.linear_fc2.weight",
        "visual.merger.norm.weight",
        "visual.merger.linear_fc1.weight",
        "visual.merger.linear_fc2.weight",
        "visual.pos_embed.weight",
        "visual.patch_embed.proj.weight",
        "visual.patch_embed.proj.bias",
    }
    got = {}
    for name, tensor in iter_mmproj_vision_weights(MMPROJ_BF16, torch.device("cpu")):
        if name in wanted:
            got[name] = tensor
    assert wanted <= got.keys()

    with safetensors.safe_open(hf_file, framework="pt", device="cpu") as f:
        for name in wanted:
            hf_tensor = f.get_tensor("model." + name)
            assert got[name].shape == hf_tensor.shape, name
            assert torch.equal(got[name], hf_tensor), name


@needs_mmproj
def test_patch_embed_temporal_stack_shape():
    tensors = dict(iter_mmproj_vision_weights(MMPROJ_BF16, torch.device("cpu")))
    w = tensors["visual.patch_embed.proj.weight"]
    assert w.shape == (1152, 3, 2, 16, 16)


# ---------------------------------------------------------------------------
# (c) discovery precedence: BF16 > F16 > any other mmproj*.gguf
# ---------------------------------------------------------------------------


def _fake_clip_arch(monkeypatch, arches: dict):
    def fake_arch(path):
        return arches[path]

    monkeypatch.setattr("freetoken.models.gguf.reader.gguf_architecture", fake_arch)


def test_discovery_prefers_bf16_over_f16(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    main = model_dir / "model-00001-of-00001.gguf"
    main.touch()
    f16 = model_dir / "mmproj-F16.gguf"
    bf16 = model_dir / "mmproj-BF16.gguf"
    f16.touch()
    bf16.touch()

    monkeypatch.setattr("freetoken.models.gguf.reader.resolve_gguf_path", lambda _p: str(main))
    _fake_clip_arch(monkeypatch, {str(f16): "clip", str(bf16): "clip"})

    assert discover_mmproj_path(str(model_dir)) == str(bf16)


def test_discovery_falls_back_to_any_clip_file(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    main = model_dir / "model-00001-of-00001.gguf"
    main.touch()
    other = model_dir / "mmproj-Q8_0.gguf"
    other.touch()

    monkeypatch.setattr("freetoken.models.gguf.reader.resolve_gguf_path", lambda _p: str(main))
    _fake_clip_arch(monkeypatch, {str(other): "clip"})

    assert discover_mmproj_path(str(model_dir)) == str(other)


def test_discovery_returns_none_without_mmproj(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    main = model_dir / "model-00001-of-00001.gguf"
    main.touch()

    monkeypatch.setattr("freetoken.models.gguf.reader.resolve_gguf_path", lambda _p: str(main))
    assert discover_mmproj_path(str(model_dir)) is None


def test_discovery_env_override(tmp_path, monkeypatch):
    override = tmp_path / "custom-mmproj.gguf"
    override.touch()
    monkeypatch.setattr("freetoken.models.gguf.reader.gguf_architecture", lambda _p: "clip")
    monkeypatch.setenv(MMPROJ_PATH_ENV, str(override))
    assert discover_mmproj_path("/irrelevant/model/path") == str(override)


def test_discovery_override_param_rejects_non_clip(monkeypatch):
    monkeypatch.setattr("freetoken.models.gguf.reader.gguf_architecture", lambda _p: "qwen4exp")
    with pytest.raises(ValueError, match="not a clip"):
        discover_mmproj_path("/irrelevant/model/path", override="/some/file.gguf")


# ---------------------------------------------------------------------------
# (d) --mmproj flag parsing
# ---------------------------------------------------------------------------


def _dummy_hf_config():
    class _Cfg:
        def to_dict(self):
            return {"torch_dtype": "bfloat16"}

    return _Cfg()


def test_mmproj_flag_defaults_to_none():
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: _dummy_hf_config()):
        args, _run_shell = parse_args(["--model", "/models/anon"])
    assert args.mm.mmproj_path is None


def test_mmproj_flag_sets_mm_mmproj_path():
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: _dummy_hf_config()):
        args, _run_shell = parse_args(
            ["--model", "/models/anon", "--mmproj", "/models/x/mmproj-BF16.gguf"]
        )
    assert args.mm.mmproj_path == "/models/x/mmproj-BF16.gguf"
