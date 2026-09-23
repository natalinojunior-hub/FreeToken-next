"""``freetoken.tuning.profile``: key composition/invalidation, schema-versioned
save/load, loader precedence, and env-default setdefault semantics. All CPU-only, fakes
only -- no server, no CUDA."""

from __future__ import annotations

from freetoken.tuning.profile import (
    SCHEMA_VERSION,
    CandidateEvidence,
    Profile,
    TunedSettings,
    apply_env_defaults,
    compute_key,
    env_overrides,
    load,
    resolve,
    save,
)


def _model_dir(tmp_path, name="weights.safetensors", size=1024):
    d = tmp_path / "model"
    d.mkdir()
    (d / name).write_bytes(b"\0" * size)
    return str(d)


def test_key_is_deterministic_for_the_same_inputs(tmp_path):
    model = _model_dir(tmp_path)
    k1 = compute_key(gpu_uuid="GPU-x", model_path=model, kv_format="turbo3", max_seq_len=16000)
    k2 = compute_key(gpu_uuid="GPU-x", model_path=model, kv_format="turbo3", max_seq_len=16000)
    assert k1 == k2


def test_key_changes_with_gpu_uuid(tmp_path):
    model = _model_dir(tmp_path)
    k1 = compute_key(gpu_uuid="GPU-a", model_path=model, kv_format="turbo3", max_seq_len=16000)
    k2 = compute_key(gpu_uuid="GPU-b", model_path=model, kv_format="turbo3", max_seq_len=16000)
    assert k1 != k2


def test_key_changes_with_kv_format(tmp_path):
    model = _model_dir(tmp_path)
    k1 = compute_key(gpu_uuid="GPU-x", model_path=model, kv_format="turbo3", max_seq_len=16000)
    k2 = compute_key(gpu_uuid="GPU-x", model_path=model, kv_format="bf16", max_seq_len=16000)
    assert k1 != k2


def test_key_context_bucket_is_ceil_pow2_not_exact_length(tmp_path):
    model = _model_dir(tmp_path)
    # 12000 and 16000 both round up to the 16384 bucket -> same key
    k1 = compute_key(gpu_uuid="GPU-x", model_path=model, kv_format="turbo3", max_seq_len=12000)
    k2 = compute_key(gpu_uuid="GPU-x", model_path=model, kv_format="turbo3", max_seq_len=16000)
    assert k1 == k2
    # 16385 rounds up to 32768 -> different key
    k3 = compute_key(gpu_uuid="GPU-x", model_path=model, kv_format="turbo3", max_seq_len=16385)
    assert k1 != k3


def test_key_changes_when_weight_file_is_touched(tmp_path):
    model = _model_dir(tmp_path)
    k1 = compute_key(gpu_uuid="GPU-x", model_path=model, kv_format="turbo3", max_seq_len=16000)
    import os
    import time

    time.sleep(0.01)
    os.utime(os.path.join(model, "weights.safetensors"), None)
    k2 = compute_key(gpu_uuid="GPU-x", model_path=model, kv_format="turbo3", max_seq_len=16000)
    assert k1 != k2


def _sample_profile(key: str) -> Profile:
    return Profile(
        key=key,
        chosen=TunedSettings(
            spec_mtp=1, defer_replay=True, draft_graph=False, moe_strategy="hybrid"
        ),
        evidence=CandidateEvidence(
            cold_pp=1000.0, committed_tg=45.0, ttft_s=0.2, peak_vram_mib=12000.0, date="2026-09-23"
        ),
        candidates=[{"settings": {"spec_mtp": 0}, "result": {}}],
    )


def test_save_load_round_trip(tmp_path):
    path = str(tmp_path / "p.json")
    prof = _sample_profile("KEY123")
    save(prof, path)
    loaded = load("KEY123", path)
    assert loaded is not None
    assert loaded.chosen == prof.chosen
    assert loaded.evidence.committed_tg == 45.0


def test_load_missing_file_returns_none(tmp_path):
    assert load("nope", str(tmp_path / "missing.json")) is None


def test_load_rejects_schema_mismatch(tmp_path):
    path = str(tmp_path / "p.json")
    prof = _sample_profile("KEY123")
    save(prof, path)
    import json

    with open(path) as f:
        raw = json.load(f)
    raw["schema"] = 999
    with open(path, "w") as f:
        json.dump(raw, f)
    assert load("KEY123", path) is None


def test_load_rejects_key_mismatch(tmp_path):
    path = str(tmp_path / "p.json")
    save(_sample_profile("KEY123"), path)
    assert load("SOME-OTHER-KEY", path) is None


def test_load_rejects_malformed_json(tmp_path):
    path = tmp_path / "p.json"
    path.write_text("{not json")
    assert load("KEY123", str(path)) is None


def test_resolve_prefers_explicit_user_value_over_tuned():
    # user explicitly set moe_strategy to "offload"; a tuned "hybrid" must not override it
    assert resolve("offload", "auto", "hybrid") == "offload"


def test_resolve_fills_sentinel_from_tuned_value():
    assert resolve("auto", "auto", "hybrid") == "hybrid"


def test_resolve_leaves_sentinel_when_no_tuned_value():
    assert resolve("auto", "auto", None) == "auto"


def test_env_overrides_maps_booleans_to_getenv_strings():
    settings = TunedSettings(
        spec_mtp=1, defer_replay=True, draft_graph=False, moe_strategy="offload"
    )
    assert env_overrides(settings) == {
        "FREETOKEN_SPEC_DEFER_REPLAY": "1",
        "FREETOKEN_DRAFT_GRAPH": "0",
    }


def test_apply_env_defaults_only_fills_unset_keys():
    settings = TunedSettings(
        spec_mtp=1, defer_replay=True, draft_graph=True, moe_strategy="offload"
    )
    environ = {"FREETOKEN_DRAFT_GRAPH": "0"}  # user explicitly disabled it
    applied = apply_env_defaults(settings, environ)
    assert applied == ["FREETOKEN_SPEC_DEFER_REPLAY"]
    assert environ["FREETOKEN_DRAFT_GRAPH"] == "0"  # untouched
    assert environ["FREETOKEN_SPEC_DEFER_REPLAY"] == "1"


def test_new_profile_carries_current_schema():
    # ft tune once hardcoded schema=1, so every profile it wrote after a bump was ignored
    assert _sample_profile("k").schema == SCHEMA_VERSION
