"""GGUF expert-format plumbing for the hybrid/offload auto-decision:

  * ``gguf_bench_key`` (cpu_executor.py): (gate_up, down) ggml types -> the same format
    string both ``ft bench bw`` (benchbw.py) and the engine's auto resolution key a
    benchbw-profile entry by.
  * ``benchbw._split_gguf_fmt`` / ``_offload_bank_specs``: the bench-side geometry for a
    uniform or mixed K-quant/I-quant format string.
  * ``bench_profile.load_backend_recommendation``: reading a profile keyed by a GGUF pair
    string back out, same as any other format.

CPU-only: format-string/geometry math and JSON-profile reads, no GPU kernels.
"""

from __future__ import annotations

import json

from freetoken.moe.benchbw import (
    DTYPE_WORKLOADS,
    _offload_bank_specs,
    _offload_cache_quant_kwargs,
    _split_gguf_fmt,
)
from freetoken.moe.bench_profile import load_backend_recommendation
from freetoken.moe.cpu_executor import dominant_gguf_pair, gguf_bench_key


def test_dominant_gguf_pair_flat_tuple():
    assert dominant_gguf_pair((21, 20)) == (21, 20)


def test_dominant_gguf_pair_per_layer_dict_majority_vote():
    types = {"gate_up": [21, 21, 23, 21, 21], "down": [20, 20, 8, 20, 20]}
    assert dominant_gguf_pair(types) == (21, 20)


def test_dominant_gguf_pair_tie_breaks_on_smaller_type_id():
    types = {"gate_up": [21, 23], "down": [20, 8]}
    assert dominant_gguf_pair(types) == (21, 8)


def test_dominant_gguf_pair_none_and_empty():
    assert dominant_gguf_pair(None) is None
    assert dominant_gguf_pair({"gate_up": [], "down": []}) is None


def test_gguf_bench_key_uniform_pair_collapses_to_one_name():
    # ggml type 21 == iq3_s on both banks
    assert gguf_bench_key(21, 21) == "iq3_s"


def test_gguf_bench_key_mixed_pair_matches_real_checkpoint():
    # Qwen3.8-Flash-Next-Unsloth-IQ4_XS: gate_up ggml type 21 (iq3_s), down type 20 (iq4_nl)
    assert gguf_bench_key(21, 20) == "iq3_s+iq4_nl"


def test_gguf_bench_key_unmapped_type_is_none():
    assert gguf_bench_key(999, 20) is None


def test_gguf_bench_key_non_mixable_pair_is_none():
    # q4_0 (ggml type 2) sits outside the K-quant/I-quant mixable family
    assert gguf_bench_key(2, 21) is None


def test_split_gguf_fmt_round_trips_bench_key():
    assert _split_gguf_fmt(gguf_bench_key(21, 20)) == ("iq3_s", "iq4_nl")
    assert _split_gguf_fmt(gguf_bench_key(21, 21)) == ("iq3_s", "iq3_s")
    assert _split_gguf_fmt("bf16") is None


def test_offload_bank_specs_mixed_pair_byte_sizes():
    # gate_up: iq3_s (256-block, 110 B/block); down: iq4_nl (32-block, 18 B/block)
    specs = _offload_bank_specs("iq3_s+iq4_nl", H=3072, I=1536)
    assert specs["gate_up"] == (2 * 1536 * (3072 // 256) * 110, __import__("torch").uint8)
    assert specs["down"] == (3072 * (1536 // 32) * 18, __import__("torch").uint8)


def test_offload_bank_specs_rejects_unaligned_dims():
    import pytest

    with pytest.raises(NotImplementedError):
        _offload_bank_specs("iq3_s+iq4_nl", H=3000, I=1536)  # not a multiple of 256


def test_real_checkpoint_pair_has_a_dtype_workload_entry():
    key = gguf_bench_key(21, 20)
    assert key in DTYPE_WORKLOADS


def test_offload_cache_quant_kwargs_routes_gguf_through_the_container_shape():
    # OffloadMoeCache.__post_init__ asserts quant_format in _BANK_SCHEMAS, which only knows
    # the "gguf" container tag -- a literal "iq3_s+iq4_nl" quant_format would trip that
    # assert (uncaught by the (ImportError, RuntimeError) handlers around it) and abort the
    # whole default `ft bench bw` run. Every GGUF format, uniform or mixed, must route
    # through "gguf" + gguf_expert_types instead.
    assert _offload_cache_quant_kwargs("iq3_s+iq4_nl") == {
        "quant_format": "gguf",
        "gguf_expert_types": (21, 20),
    }
    assert _offload_cache_quant_kwargs("iq3_s") == {
        "quant_format": "gguf",
        "gguf_expert_types": (21, 21),
    }


def test_offload_cache_quant_kwargs_non_gguf_passes_through():
    assert _offload_cache_quant_kwargs("bf16") == {"quant_format": "bf16"}
    assert _offload_cache_quant_kwargs("nvfp4") == {"quant_format": "nvfp4"}


def test_resolve_gguf_format_handles_per_layer_list_shape():
    # Pre-existing crash fix: OffloadMoeCache normalizes gguf_expert_types to a per-layer
    # list[tuple[int, int]] (offload_cache.py __post_init__); _resolve_gguf_format used to
    # index it as a flat (gate_up, down) pair (int(types[0]) on a tuple -> TypeError) for
    # any checkpoint whose banks reach it this way -- every real GGUF checkpoint on
    # --moe-strategy cpu/hybrid, not just a mixed one.
    from types import SimpleNamespace

    from freetoken.moe.cpu_executor import _resolve_gguf_format

    cache = SimpleNamespace(gguf_expert_types=[(21, 20), (23, 20), (21, 8)])
    assert _resolve_gguf_format(cache) == ("iq3_s", "iq4_nl")  # dominant pair
    cache_uniform = SimpleNamespace(gguf_expert_types=(21, 20))
    assert _resolve_gguf_format(cache_uniform) == ("iq3_s", "iq4_nl")


def test_profile_lookup_matches_gguf_pair_key(tmp_path):
    path = tmp_path / "benchbw.json"
    path.write_text(
        json.dumps({"gpu": {"name": "FAKE GPU"}, "dtypes": {"iq3_s+iq4_nl": "hybrid"}})
    )
    key = gguf_bench_key(21, 20)
    assert load_backend_recommendation(key, gpu_name="FAKE GPU", path=str(path)) == "hybrid"
    # a bare "gguf" container tag (unresolved) finds no entry -> safe default (offload)
    assert load_backend_recommendation("gguf", gpu_name="FAKE GPU", path=str(path)) is None
