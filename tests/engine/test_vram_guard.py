from types import SimpleNamespace

import torch

from freetoken.engine.engine import Engine

MIB = 1 << 20


def _engine(monkeypatch, free, peak, now):
    eng = Engine.__new__(Engine)
    eng.device = SimpleNamespace(type="cuda")
    eng.moe_offload_cache = SimpleNamespace(cache_size=1000)
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
