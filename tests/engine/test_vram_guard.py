from types import SimpleNamespace

import torch

from freetoken.engine.engine import Engine

MIB = 1 << 20


def _engine(monkeypatch, free, peak, now):
    eng = Engine.__new__(Engine)
    eng.device = SimpleNamespace(type="cuda")
    eng.moe_offload_cache = SimpleNamespace(cache_size=1000, resident_rows=1000)
    eng._vram_guard_armed = True
    calls = []
    eng._target_moe_and_expert_bytes = lambda _: (None, 10 * MIB)
    eng.rebuild_runtime_cache = lambda moe_cache_size: calls.append(moe_cache_size)
    eng._charge_expert_cache = lambda cache: None
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d: (free, 16 << 30))
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda d: peak)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d: now)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda d: None)
    return eng, calls


def test_guard_shrinks_experts_when_peak_left_too_little(monkeypatch):
    # 300 MiB free now, but the peak held 200 MiB more than now -> 100 MiB free at the peak
    eng, calls = _engine(monkeypatch, free=300 * MIB, peak=1200 * MIB, now=1000 * MIB)
    eng.guard_vram_at_idle()
    assert calls == [1000 - 16]  # ceil((256 - 100) MiB / 10 MiB) slots given back


def test_guard_keeps_cache_when_peak_had_margin(monkeypatch):
    eng, calls = _engine(monkeypatch, free=900 * MIB, peak=1200 * MIB, now=1000 * MIB)
    eng.guard_vram_at_idle()
    assert calls == []


def test_guard_regrows_after_calm_windows_up_to_plan(monkeypatch):
    # 2 GiB free at every peak: grows only on the 3rd calm window, capped at the startup plan
    eng, calls = _engine(monkeypatch, free=2048 * MIB, peak=1000 * MIB, now=1000 * MIB)
    eng._expert_plan_slots = 1040
    for _ in range(2):
        eng.guard_vram_at_idle()
    assert calls == []
    eng.guard_vram_at_idle()
    assert calls == [1040]


def test_guard_never_grows_past_plan(monkeypatch):
    eng, calls = _engine(monkeypatch, free=4096 * MIB, peak=1000 * MIB, now=1000 * MIB)
    eng._expert_plan_slots = 1000
    for _ in range(5):
        eng.guard_vram_at_idle()
    assert calls == []


def _residency_engine(monkeypatch, free):
    eng, calls = _engine(monkeypatch, free=free, peak=1000 * MIB, now=1000 * MIB)
    eng._resize_experts = calls.append
    eng.moe_offload_cache = SimpleNamespace(
        cache_size=2000,  # VMM shape
        resident_rows=1000,
        pools=[SimpleNamespace(layers=[0, 1])],
        num_experts=1000,
        _vmm_arenas=[object()],
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return eng, calls


def test_decode_residency_grows_into_measured_free_and_folds_back(monkeypatch):
    # 2 GiB free, reserve = 256 MiB margin + one 2 MiB granule per region -> 179 slots of 10 MiB
    eng, calls = _residency_engine(monkeypatch, free=2048 * MIB)
    eng.set_decode_residency(True)
    assert calls == [1179]
    eng.moe_offload_cache.resident_rows = 1179
    eng.set_decode_residency(True)  # already grown: no churn
    eng.set_decode_residency(False)
    assert calls[1:] == [1000]
    eng.set_decode_residency(False)
    assert len(calls) == 2


def test_decode_residency_never_touches_the_reserve(monkeypatch):
    eng, calls = _residency_engine(monkeypatch, free=258 * MIB)
    eng.set_decode_residency(True)
    assert calls == [] and eng._expert_decode_slots is None


def test_decode_residency_failed_grow_restores_prefill_size_and_learns(monkeypatch):
    eng, calls = _residency_engine(monkeypatch, free=2048 * MIB)

    def resize(target):
        calls.append(target)
        if target > 1000:
            raise RuntimeError("cuMemCreate failed")

    eng._resize_experts = resize
    eng.set_decode_residency(True)
    assert calls == [1179, 1000] and eng._expert_decode_slots is None
    assert eng._decode_reserve_learned == 179 * 10 * MIB
