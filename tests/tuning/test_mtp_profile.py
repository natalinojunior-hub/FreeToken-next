"""``freetoken.tuning.mtp_profile``: fingerprint composition/invalidation, schema-versioned
save/load, the FREETOKEN_MTP_PROFILE modes (auto/off/refresh), and the calibration-source
versioning that forces recalibration when depth-deciding code changes. CPU-only, fakes only."""

from __future__ import annotations

import json

import pytest

from freetoken.tuning import mtp_profile as mp


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """Pin the cache dir, GPU uuid and source fingerprint so keys are deterministic and the
    real ~/.cache and source tree are never touched."""
    monkeypatch.setattr(mp, "_cache_dir", lambda: str(tmp_path / "cache"))
    monkeypatch.setattr(mp, "_gpu_uuid", lambda: "GPU-test")
    monkeypatch.setattr(mp, "_calibration_source_fingerprint", lambda: "srcfp0000000000")
    monkeypatch.delenv("FREETOKEN_MTP_PROFILE", raising=False)
    model = tmp_path / "model"
    model.mkdir()
    (model / "w.gguf").write_bytes(b"\0" * 2048)
    return str(model)


def _key(model, **over):
    base = dict(
        model_path=model,
        kv_format="auto",
        max_seq_len=16704,
        spec_mtp_cap=4,
        max_running_req=1,
        moe_strategy="offload",
        compact_state=True,
        ssm_dtype="bfloat16",
        kv_tiering="off",
        kv_ram_tokens=0,
        kv_reserve_tokens=0,
    )
    base.update(over)
    return mp.compute_key(**base)


def test_key_deterministic(isolated):
    assert _key(isolated) == _key(isolated)


def test_source_fingerprint_includes_state_pool_implementation(monkeypatch, tmp_path):
    package = tmp_path / "freetoken"
    state_dir = package / "kvcache"
    state_dir.mkdir(parents=True)
    source = state_dir / "linear_state_pool.py"
    monkeypatch.setattr(mp, "__file__", str(package / "tuning" / "mtp_profile.py"))
    source.write_text("slots = 9\n")
    fingerprint = mp._calibration_source_fingerprint.__wrapped__()
    source.write_text("slots = 7\n")
    assert fingerprint != mp._calibration_source_fingerprint.__wrapped__()


@pytest.mark.parametrize(
    "axis,value",
    [
        ("kv_format", "fp8"),
        ("spec_mtp_cap", 2),
        ("max_running_req", 2),
        ("moe_strategy", "hybrid"),
        ("compact_state", False),
        ("ssm_dtype", "float32"),
        ("kv_tiering", "auto"),
        ("kv_ram_tokens", 8192),
        ("kv_reserve_tokens", 4096),
        ("max_seq_len", 262144),  # a different ceil-pow2 context bucket
    ],
)
def test_key_changes_with_each_config_axis(isolated, axis, value):
    assert _key(isolated) != _key(isolated, **{axis: value})


def test_key_changes_with_gpu_and_source_and_model(isolated, monkeypatch, tmp_path):
    base = _key(isolated)
    monkeypatch.setattr(mp, "_gpu_uuid", lambda: "GPU-other")
    assert base != _key(isolated)
    monkeypatch.setattr(mp, "_gpu_uuid", lambda: "GPU-test")
    # a changed calibration-source fingerprint (depth-deciding code edited) invalidates
    monkeypatch.setattr(mp, "_calibration_source_fingerprint", lambda: "srcfp1111111111")
    assert base != _key(isolated)
    monkeypatch.setattr(mp, "_calibration_source_fingerprint", lambda: "srcfp0000000000")
    # a different model (weight identity) invalidates
    other = tmp_path / "other"
    other.mkdir()
    (other / "w.gguf").write_bytes(b"\1" * 4096)
    assert base != _key(other)


def test_save_load_roundtrip(isolated):
    k = _key(isolated)
    assert mp.load(k) is None  # cold: nothing cached
    assert mp.save(k, 4) is not None
    assert mp.load(k) == 4  # warm: the learned depth comes back


def test_load_rejects_schema_and_key_mismatch(isolated):
    k = _key(isolated)
    mp.save(k, 4)
    path = mp.profile_path(k)
    raw = json.loads(open(path).read())
    raw["schema"] = mp.SCHEMA_VERSION + 1  # foreign schema
    open(path, "w").write(json.dumps(raw))
    assert mp.load(k) is None
    raw["schema"] = mp.SCHEMA_VERSION
    raw["key"] = "some-other-key"  # hand-copied/renamed file
    open(path, "w").write(json.dumps(raw))
    assert mp.load(k) is None


def test_load_rejects_malformed_depth(isolated):
    k = _key(isolated)
    mp.save(k, 4)
    path = mp.profile_path(k)
    raw = json.loads(open(path).read())
    raw["depth"] = "four"  # not an int
    open(path, "w").write(json.dumps(raw))
    assert mp.load(k) is None


def test_mode_off_neither_loads_nor_saves(isolated, monkeypatch):
    k = _key(isolated)
    mp.save(k, 4)  # cached while auto
    monkeypatch.setenv("FREETOKEN_MTP_PROFILE", "off")
    assert mp.load(k) is None  # off ignores the cache
    assert mp.save(k, 3) is None  # off never writes
    monkeypatch.delenv("FREETOKEN_MTP_PROFILE")
    assert mp.load(k) == 4  # the auto-mode entry is untouched


def test_mode_refresh_ignores_cache_but_resaves(isolated, monkeypatch):
    k = _key(isolated)
    mp.save(k, 4)
    monkeypatch.setenv("FREETOKEN_MTP_PROFILE", "refresh")
    assert mp.load(k) is None  # forced recalibration: ignore the old depth
    assert mp.save(k, 3) is not None  # but still persist the freshly learned one
    monkeypatch.delenv("FREETOKEN_MTP_PROFILE")
    assert mp.load(k) == 3  # overwritten


def test_unknown_mode_falls_back_to_auto(isolated, monkeypatch):
    k = _key(isolated)
    monkeypatch.setenv("FREETOKEN_MTP_PROFILE", "bogus")
    assert mp._profile_mode() == "auto"
    assert mp.save(k, 4) is not None
    assert mp.load(k) == 4
