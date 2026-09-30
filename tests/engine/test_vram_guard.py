from contextlib import nullcontext
from types import SimpleNamespace

import torch
import pytest

from freetoken.engine.engine import Engine

MIB = 1 << 20


def _engine(monkeypatch, free, peak, now):
    eng = Engine.__new__(Engine)
    eng.device = SimpleNamespace(type="cuda")
    eng.moe_offload_cache = SimpleNamespace(cache_size=1000, resident_rows=1000)
    eng._vram_guard_armed = True
    eng._prefill_transient_reserve = 256 * MIB
    eng._decode_reserve_learned = 0
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
    eng._decode_reserve_learned = 256 * MIB
    eng.guard_vram_at_idle()
    assert calls == [1000 - 16]  # ceil((256 - 100) MiB / 10 MiB) slots given back


def test_guard_keeps_cache_when_peak_had_margin(monkeypatch):
    eng, calls = _engine(monkeypatch, free=900 * MIB, peak=1200 * MIB, now=1000 * MIB)
    eng.guard_vram_at_idle()
    assert calls == []


def test_guard_regrows_after_calm_windows_to_measured_headroom(monkeypatch):
    # A stale startup split must not cap growth when the reserved shape and measured headroom allow it.
    eng, calls = _engine(monkeypatch, free=2048 * MIB, peak=1000 * MIB, now=1000 * MIB)
    eng._decode_reserve_learned = 256 * MIB
    eng.moe_offload_cache.cache_size = 1200
    eng._expert_plan_slots = 1040
    for _ in range(2):
        eng.guard_vram_at_idle()
    assert calls == []
    eng.guard_vram_at_idle()
    assert calls == [1179]


def test_guard_never_grows_past_reserved_shape(monkeypatch):
    eng, calls = _engine(monkeypatch, free=4096 * MIB, peak=1000 * MIB, now=1000 * MIB)
    eng.moe_offload_cache.cache_size = 1000
    eng._expert_plan_slots = 900
    for _ in range(5):
        eng.guard_vram_at_idle()
    assert calls == []


def test_guard_does_not_charge_successful_prefill_transient_twice(monkeypatch):
    eng, calls = _engine(monkeypatch, free=100 * MIB, peak=1200 * MIB, now=1000 * MIB)
    eng._prefill_transient_reserve = 2 << 30
    eng._decode_reserve_learned = 64 * MIB
    eng._guard_window(100 * MIB)
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
        expert_pool_bytes=1000 * 10 * MIB,
        backed_bytes_for=lambda size: size * 10 * MIB,
    )
    eng._decode_reserve_learned = 0
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return eng, calls


def test_decode_residency_grows_into_measured_free_and_folds_back(monkeypatch):
    # Exact shaped backing is admitted up to the measured free-byte boundary.
    eng, calls = _residency_engine(monkeypatch, free=2048 * MIB)
    eng.set_decode_residency(True)
    assert eng._decode_allocator_baseline == 1000 * MIB
    assert calls == [1204]
    eng.moe_offload_cache.resident_rows = 1204
    eng.set_decode_residency(True)  # already grown: no churn
    eng.set_decode_residency(False)
    assert calls[1:] == [1000]
    eng.set_decode_residency(False)
    assert len(calls) == 2


def test_decode_residency_never_touches_the_reserve(monkeypatch):
    # The measured 1024 MiB admits the largest exact 10 MiB-row prefix.
    eng, calls = _residency_engine(monkeypatch, free=1024 * MIB)
    syncs = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: syncs.append(None))
    eng.set_decode_residency(True)
    assert calls == [1102] and eng._expert_decode_slots == 1102
    eng.moe_offload_cache.resident_rows = 1102
    eng.set_decode_residency(True)
    assert calls == [1102] and len(syncs) == 1  # the grown phase is not retried each decode


def test_drained_scheduler_transition_skips_device_wide_sync(monkeypatch):
    eng, calls = _residency_engine(monkeypatch, free=1024 * MIB)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda _device: pytest.fail("scheduler already drained the engine stream"),
    )
    eng.set_decode_residency(True, stream_drained=True)
    assert calls == [1102]


def test_decode_residency_failed_grow_does_not_keep_external_pressure_debt(monkeypatch):
    eng, calls = _residency_engine(monkeypatch, free=2048 * MIB)
    cache = eng.moe_offload_cache
    refused = [True]

    def resize(target):
        calls.append(target)
        if target > 1000 and refused[0]:
            refused[0] = False  # one-off external backing refusal; reserved bytes do not change
            raise RuntimeError("cuMemCreate failed")
        cache.resident_rows = target

    eng._resize_experts = resize
    eng.set_decode_residency(True)
    assert calls == [1204] and eng._expert_decode_slots == 1000
    assert eng._decode_reserve_learned == 0
    assert eng._decode_allocator_baseline == 1000 * MIB

    eng.set_decode_residency(False)  # close the failed decode phase before the next request
    eng.set_decode_residency(True)
    assert calls == [1204, 1000, 1204]
    assert cache.resident_rows == 1204
    assert eng._decode_reserve_learned == 0


def test_decode_residency_scans_exact_nonmonotone_pool_costs(monkeypatch):
    eng, calls = _residency_engine(monkeypatch, free=256 * MIB)
    cache = eng.moe_offload_cache
    cache.cache_size = 212
    cache.resident_rows = 210
    cache.expert_pool_bytes = 921 * MIB

    def backing_bytes(rows):
        # Geometry floors transfer rows between pools at 212: 211 costs 931 MiB, while
        # 212 costs 626 MiB. A monotone binary search could miss the larger fitting target.
        return {211: 931 * MIB, 212: 626 * MIB}.get(rows, 2_000 * MIB)

    cache.backed_bytes_for = backing_bytes
    eng.set_decode_residency(True)
    assert calls == [212]


def test_decode_residency_keeps_forward_reserve_with_exact_vmm_price(monkeypatch):
    eng, calls = _residency_engine(monkeypatch, free=1024 * MIB)
    eng._decode_reserve_learned = 64 * MIB
    eng.moe_offload_cache._vmm_arenas = [SimpleNamespace(g=8 * MIB)]
    eng.moe_offload_cache.pools = [object(), object()]

    eng.set_decode_residency(True)

    # 64 MiB measured forward reserve exceeds two native 8 MiB backing granules.
    assert calls == [1096]


def _vmm_guard_engine(monkeypatch, free, arenas=2, pools=10):
    eng, calls = _engine(monkeypatch, free=free, peak=1000 * MIB, now=1000 * MIB)
    eng.moe_offload_cache = SimpleNamespace(
        cache_size=2000,
        resident_rows=1000,
        pools=[object()] * pools,
        _vmm_arenas=[SimpleNamespace(g=2 * MIB) for _ in range(arenas)],
        live_caps=[100] * pools,
        set_live=lambda n: calls.append(("set_live", n)),
    )
    eng._resize_experts = calls.append  # bypass the real method's stream/VMM plumbing
    eng._expert_plan_slots = 1200
    return eng, calls


def test_guard_regrow_prices_the_vmm_granule_rounding(monkeypatch):
    # Free at peak minus the measured reserve and native VMM slack is spendable.
    eng, calls = _vmm_guard_engine(monkeypatch, free=2048 * MIB)
    eng._decode_reserve_learned = 256 * MIB
    for _ in range(3):
        eng.guard_vram_at_idle()
    assert calls == [1000 + (2048 - 256 - 40) * MIB // (10 * MIB)]


def test_guard_regrow_backing_failure_charges_safe_prefix_without_learning_debt(monkeypatch):
    eng, calls = _vmm_guard_engine(monkeypatch, free=2048 * MIB)
    eng._decode_reserve_learned = 256 * MIB
    cache = eng.moe_offload_cache
    charged = []

    def boom(target):
        cache.resident_rows = 950
        cache.live_caps = [95] * len(cache.live_caps)
        raise RuntimeError("cuMemCreate failed with CUresult 2")

    eng._resize_experts = boom
    eng._charge_expert_cache = lambda actual: charged.append(actual.resident_rows)
    for _ in range(3):
        eng.guard_vram_at_idle()  # must not raise out of the idle hook
    assert eng._decode_reserve_learned == 256 * MIB  # no allocator growth was measured
    assert ("set_live", 1000) not in calls  # set_live owns its own allocation-free rollback
    assert cache.resident_rows == 950
    assert charged == [950]

    def recover(target):
        calls.append(target)
        cache.resident_rows = target

    eng._resize_experts = recover
    for _ in range(3):
        eng.guard_vram_at_idle()
    assert calls[-1] == 1125  # exact available backing after transient refusal
    assert cache.resident_rows == 1125
    assert eng._decode_reserve_learned == 256 * MIB


def test_guard_shrink_failure_stays_fatal(monkeypatch):
    eng, calls = _vmm_guard_engine(monkeypatch, free=100 * MIB)
    eng._decode_reserve_learned = 256 * MIB

    def boom(target):
        raise RuntimeError("backing refused")

    eng._resize_experts = boom
    import pytest

    with pytest.raises(RuntimeError):
        eng.guard_vram_at_idle()


def _pressure_engine(monkeypatch, *, rows=1000, floor=1):
    eng = Engine.__new__(Engine)
    eng.device = SimpleNamespace(type="cuda")
    eng.config = SimpleNamespace(tp_info=SimpleNamespace(size=1))
    eng.stream = object()
    calls = []

    def set_live(size):
        actual = max(floor, size)
        calls.append(("set_live", size, actual))
        cache.resident_rows = actual
        cache.live_caps = [actual]
        return actual

    cache = SimpleNamespace(
        cache_size=1200,
        resident_rows=rows,
        live_caps=[rows],
        pools=[object()],
        _vmm_arenas=[SimpleNamespace(g=2 * MIB)],
        set_live=set_live,
    )
    eng.moe_offload_cache = cache
    eng._target_moe_and_expert_bytes = lambda _: (None, 10 * MIB)
    eng._charge_expert_cache = lambda _cache: None
    eng.rebuild_runtime_cache = lambda **kwargs: calls.append(("rebuild", kwargs))
    eng._expert_decode_slots = rows + 10
    eng._expert_prefill_slots = rows
    eng._decode_reserve_learned = 256 * MIB
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    return eng, cache, calls


def test_before_forward_guard_syncs_then_shrinks_vmm_and_clamps_foldback(monkeypatch):
    eng, cache, calls = _pressure_engine(monkeypatch, rows=1000)
    events = []

    def driver_free(_device):
        # The mocked VMM shrink returns enough physical memory to leave the margin.
        return (100 if cache.resident_rows == 1000 else 300) * MIB, 16 << 30

    monkeypatch.setattr(torch.cuda, "mem_get_info", driver_free)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: events.append("sync"))
    original_set_live = cache.set_live

    def record_set_live(size):
        events.append("set_live")
        return original_set_live(size)

    cache.set_live = record_set_live
    assert eng.guard_vram_before_forward() is False
    assert events.index("sync") < events.index("set_live")
    assert cache.resident_rows == cache.live_caps[0] < 1000
    assert eng._expert_decode_slots == cache.resident_rows
    assert eng._expert_prefill_slots == cache.resident_rows
    eng.set_decode_residency(False)
    assert calls[-1] == ("set_live", cache.resident_rows, cache.resident_rows)
    assert not any(call[0] == "rebuild" for call in calls)


def test_before_forward_guard_calm_path_does_not_sync_or_resize(monkeypatch):
    eng, cache, calls = _pressure_engine(monkeypatch)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (300 * MIB, 16 << 30))
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda _device: pytest.fail("calm guard must not synchronize"),
    )
    assert eng.guard_vram_before_forward() is False
    assert cache.resident_rows == 1000
    assert calls == []


def test_before_forward_guard_floor_no_progress_reports_unresolved(monkeypatch):
    eng, cache, calls = _pressure_engine(monkeypatch, rows=5, floor=5)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (0, 16 << 30))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    assert eng.guard_vram_before_forward() is True
    assert cache.resident_rows == 5
    assert calls == [("set_live", 1, 5)]


def test_before_forward_guard_learns_decode_allocator_growth(monkeypatch):
    eng, _cache, _calls = _pressure_engine(monkeypatch)
    eng._decode_reserve_learned = 0
    eng._decode_allocator_baseline = 1000 * MIB
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (1 * MIB, 16 << 30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 1064 * MIB)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 1064 * MIB)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)

    assert eng.guard_vram_before_forward(prefill=False) is True
    assert eng._decode_reserve_learned == 64 * MIB


def test_before_forward_external_pressure_does_not_learn_decode_reserve(monkeypatch):
    eng, _cache, _calls = _pressure_engine(monkeypatch)
    eng._decode_reserve_learned = 0
    eng._prefill_transient_reserve = 256 * MIB
    eng._decode_allocator_baseline = 1000 * MIB
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (100 * MIB, 16 << 30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 1000 * MIB)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 1000 * MIB)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)

    assert eng.guard_vram_before_forward(prefill=True) is True
    assert eng._decode_reserve_learned == 0


def test_decode_phase_learns_allocator_peak_without_oom(monkeypatch):
    eng = Engine.__new__(Engine)
    eng.device = SimpleNamespace(type="cuda")
    eng._decode_allocator_baseline = 1000 * MIB
    eng._decode_reserve_learned = 0
    eng._vram_profile_key = None
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 1128 * MIB)

    assert eng._learn_decode_peak() == 128 * MIB
    assert eng._decode_reserve_learned == 128 * MIB


def test_before_forward_guard_fixed_back_cache_reports_unresolved(monkeypatch):
    eng, cache, calls = _pressure_engine(monkeypatch)
    cache._vmm_arenas = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (100 * MIB, 16 << 30))
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda _device: pytest.fail("fixed-back cache cannot be resized in the live request"),
    )
    assert eng.guard_vram_before_forward() is True
    assert cache.resident_rows == 1000
    assert calls == []


def test_before_forward_prefill_reserve_shrinks_to_measured_requirement(monkeypatch):
    eng, cache, calls = _pressure_engine(monkeypatch)
    eng._prefill_transient_reserve = 1 << 30
    free = iter((300 * MIB, 300 * MIB, 1300 * MIB, 1300 * MIB))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (next(free), 16 << 30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    assert eng.guard_vram_before_forward(prefill=True) is False
    # The 1 GiB measured transient and VMM granule determine the release size.
    assert calls == [("set_live", 927, 927)]
    assert cache.resident_rows == 927


@pytest.mark.parametrize("prefill,reserved", [(True, 1 << 30), (False, 0)])
def test_before_forward_guard_skips_funded_prefill_reserve_or_decode(
    monkeypatch, prefill, reserved
):
    eng, cache, calls = _pressure_engine(monkeypatch)
    eng._prefill_transient_reserve = 1 << 30
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (300 * MIB, 16 << 30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: reserved)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 0)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda _device: pytest.fail("funded reserve or decode phase must not shrink"),
    )
    assert eng.guard_vram_before_forward(prefill=prefill) is False
    assert cache.resident_rows == 1000
    assert calls == []
