import json
from types import SimpleNamespace

import torch

from freetoken.engine.engine import Engine
from freetoken.scheduler.scheduler import Scheduler
from freetoken.tuning import vram_profile

MIB = 1 << 20


def _oom(text):
    return torch.OutOfMemoryError(text)


def test_note_oom_folds_attempted_bytes_into_reserve():
    eng = SimpleNamespace(_decode_reserve_learned=0, _vram_profile_key=None)
    attempted = Engine.note_decode_oom(eng, _oom("CUDA OOM. Tried to allocate 64.00 MiB."))
    assert attempted == 64 * MIB
    assert eng._decode_reserve_learned == attempted


def test_note_oom_never_shrinks_and_ignores_unparseable():
    eng = SimpleNamespace(_decode_reserve_learned=100 * MIB, _vram_profile_key=None)
    assert Engine.note_decode_oom(eng, _oom("Tried to allocate 8.00 MiB")) == 8 * MIB
    assert eng._decode_reserve_learned == 100 * MIB
    assert Engine.note_decode_oom(eng, _oom("no number here")) == 0
    assert eng._decode_reserve_learned == 100 * MIB


def test_note_oom_persists_and_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setattr(vram_profile, "_path", lambda key: str(tmp_path / f"{key}.json"))
    eng = SimpleNamespace(_decode_reserve_learned=0, _vram_profile_key="k1")
    Engine.note_decode_oom(eng, _oom("Tried to allocate 2.00 GiB"))
    stored = json.loads((tmp_path / "k1.json").read_text())
    assert stored["schema"] == "vram1" and stored["learned_bytes"] == eng._decode_reserve_learned
    assert vram_profile.load("k1") == eng._decode_reserve_learned


def test_compute_key_distinguishes_vision_and_mtp_axes(monkeypatch):
    monkeypatch.setattr(vram_profile, "key_from_config", lambda *a, **k: "base")
    cfg = SimpleNamespace()
    k = vram_profile.compute_key(cfg, 262144, 4, has_vision=True, has_mtp_head=False)
    j = vram_profile.compute_key(cfg, 262144, 4, has_vision=True, has_mtp_head=True)
    assert k != j
    assert k == vram_profile.compute_key(cfg, 262144, 4, has_vision=True, has_mtp_head=False)


def test_scheduler_seed_warms_engine_from_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(vram_profile, "key_from_config", lambda *a, **k: "base")
    monkeypatch.setattr(vram_profile, "_path", lambda key: str(tmp_path / f"{key}.json"))
    vram_profile.save("x", 512 * MIB)
    eng = SimpleNamespace(max_seq_len=262144, _decode_reserve_learned=0, _vram_profile_key=None)
    sched = SimpleNamespace(
        engine=eng,
        config=SimpleNamespace(
            max_seq_len_override=None,
            max_seq_len=0,
            spec_mtp=4,
            active_encoders=[],
            model_config=SimpleNamespace(mtp_layer_id=None),
        ),
    )
    # find the saved file under its computed key, not "x"
    key = vram_profile.compute_key(sched.config, 262144, 4, False, False)
    vram_profile.save(key, 512 * MIB)
    Scheduler._seed_vram_learned(sched)
    assert eng._vram_profile_key == key
    assert eng._decode_reserve_learned == 512 * MIB


def test_calibration_requires_two_matching_samples_and_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(vram_profile, "_path", lambda key: str(tmp_path / f"{key}.json"))
    key = "calibration-key"
    calibration = SimpleNamespace(
        chunk_lo=256,
        transient_lo=8 * MIB,
        chunk_hi=1024,
        transient_hi=16 * MIB,
        lazy_persistent=2 * MIB,
        graph_capture_peak=32 * MIB,
        graph_pool_size=4 * MIB,
        non_pytorch_growth=1 * MIB,
    )
    geometry = {"driver_total": 16 * 1024 * MIB, "page_size": 64, "attention_backend": "triton"}
    vram_profile.save_calibration(key, calibration, geometry=geometry)
    assert vram_profile.load_calibration(
        key, driver_total=geometry["driver_total"], page_size=64, attention_backend="triton"
    ) is None
    vram_profile.save_calibration(key, calibration, geometry=geometry)
    assert vram_profile.load_calibration(
        key, driver_total=geometry["driver_total"], page_size=64, attention_backend="triton"
    ) == vars(calibration)
    assert vram_profile.load_calibration(
        key, driver_total=geometry["driver_total"] + 512 * MIB, page_size=64, attention_backend="triton"
    ) is None


def test_calibration_change_resets_sample_hysteresis(tmp_path, monkeypatch):
    monkeypatch.setattr(vram_profile, "_path", lambda key: str(tmp_path / f"{key}.json"))
    key = "calibration-key"
    base = SimpleNamespace(
        chunk_lo=256,
        transient_lo=8 * MIB,
        chunk_hi=1024,
        transient_hi=16 * MIB,
        lazy_persistent=2 * MIB,
        graph_capture_peak=32 * MIB,
        graph_pool_size=4 * MIB,
        non_pytorch_growth=1 * MIB,
    )
    changed = SimpleNamespace(**{**vars(base), "transient_hi": 64 * MIB})
    geometry = {"driver_total": 16 * 1024 * MIB, "page_size": 64, "attention_backend": "triton"}
    vram_profile.save_calibration(key, base, geometry=geometry)
    vram_profile.save_calibration(key, base, geometry=geometry)
    assert vram_profile.load_calibration(key, driver_total=geometry["driver_total"], page_size=64, attention_backend="triton")
    vram_profile.save_calibration(key, changed, geometry=geometry)
    assert vram_profile.load_calibration(key, driver_total=geometry["driver_total"], page_size=64, attention_backend="triton") is None
