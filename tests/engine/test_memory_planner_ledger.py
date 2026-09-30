"""Canonical VRAM ledger arithmetic of MemoryPlanner (no GPU).

Covers: one ledger term per owner, transient = max(prefill, graph capture) not
their sum, context floor always funded, experts take the residual, exact
infeasibility report, and the two-point transient model.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.engine.memory_planner import (
    MemoryPlanner,
    RuntimeCalibration,
    StaticCostModel,
    _linear_pool_num_slots,
)
from freetoken.kvcache.linear_state_pool import _linear_pool_min_slots

MIB = 1 << 20


def _static(**kw) -> StaticCostModel:
    base = dict(
        weights_bytes=0,
        expert_bytes_per_slot=6 * MIB,
        kv_bytes_per_page=MIB // 4,
        kv_fixed_bytes=MIB,
        page_tokens=64,
        dummy_page_bytes=0,
        gdn_state_bytes_per_slot=100 * MIB,
        gdn_num_slots=9,
        page_table_bytes=MIB,
        expert_auxiliary_bytes=0,
        quantization_side_tables=0,
        staging_buffers=0,
        attention_backend_fixed=0,
        min_expert_slots=512,
        max_expert_slots=24576,
        total_experts=512,
        prefill_overlap=False,
    )
    base.update(kw)
    return StaticCostModel(**base)


def _calib(**kw) -> RuntimeCalibration:
    base = dict(
        chunk_lo=4096,
        transient_lo=800 * MIB,
        chunk_hi=8192,
        transient_hi=1700 * MIB,
        lazy_persistent=100 * MIB,
        graph_capture_peak=0,
        graph_pool_size=0,
        non_pytorch_growth=0,
    )
    base.update(kw)
    return RuntimeCalibration(**base)


class _Cfg:
    max_seq_len = 16384
    max_extend_tokens = 8192


def _planner(static=None, calib=None) -> MemoryPlanner:
    p = MemoryPlanner.__new__(MemoryPlanner)
    p.static_model = static or _static()
    p.runtime_calibration = calib or _calib()
    return p


def test_transient_is_linear_between_measured_points_and_floored_below():
    c = _calib()
    assert c.transient_at(8192) == 1700 * MIB
    assert c.transient_at(4096) == 800 * MIB
    assert c.transient_at(1024) == 800 * MIB
    assert c.transient_at(6144) == 1250 * MIB
    with pytest.raises(AssertionError):
        c.transient_at(16384)  # never extrapolate past what was measured


def test_phase_d_probe_reduces_chunk_below_256_after_measured_oom(monkeypatch):
    from types import SimpleNamespace

    planner = MemoryPlanner.__new__(MemoryPlanner)
    planner.device = torch.device("cuda")
    planner._probe_kv_pool = object()
    observed = []

    def prefill(_model, _config, chunk, _pool):
        observed.append(chunk)
        if chunk >= 256:
            raise torch.cuda.OutOfMemoryError("Tried to allocate 64.00 MiB")

    planner._run_validation_prefill = prefill
    planner._measure_prefill_transient = lambda model, config, chunk: (
        prefill(model, config, chunk, planner._probe_kv_pool) or chunk * MIB
    )
    planner._measure_graph_capture = lambda _config, _model: (0, 0)
    planner._trim_probe_experts_for_scratch = lambda _oom: pytest.fail(
        "a smaller measured chunk should fit before expert trimming"
    )
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_args: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_args: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_args: (8 * MIB, 16 * MIB))

    calibration = planner.phase_d_runtime_calibration(
        SimpleNamespace(max_extend_tokens=8192, max_seq_len=16384), object()
    )
    assert (calibration.chunk_lo, calibration.chunk_hi) == (64, 128)
    assert min(observed) < 256


def test_probe_trim_explains_non_vmm_cache_rejection():
    from types import SimpleNamespace

    planner = MemoryPlanner.__new__(MemoryPlanner)
    planner._probe_expert_cache = SimpleNamespace(_vmm_arenas=[])
    assert not planner._trim_probe_experts_for_scratch(
        torch.cuda.OutOfMemoryError("Tried to allocate 64.00 MiB")
    )


def test_probe_trim_releases_exact_vmm_rows_for_observed_allocation(monkeypatch):
    from types import SimpleNamespace

    targets = []
    cache = SimpleNamespace(
        _vmm_arenas=[object()],
        resident_rows=2048,
        set_live=targets.append,
    )
    planner = MemoryPlanner.__new__(MemoryPlanner)
    planner.config = SimpleNamespace(tp_info=SimpleNamespace(size=1))
    planner.model_config = SimpleNamespace(num_experts=4096)
    planner.static_model = SimpleNamespace(expert_bytes_for_slots=lambda _count: MIB)
    planner.min_pool_rows = lambda _num_experts: 1024
    planner._probe_expert_cache = cache
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    assert planner._trim_probe_experts_for_scratch(
        torch.cuda.OutOfMemoryError("Tried to allocate 64.00 MiB")
    )
    assert targets == [1984]


def test_speculative_state_is_charged_once_across_request_slots():
    from dataclasses import replace

    sm = _static(gdn_fixed_state_bytes=220 * MIB)
    assert sm.gdn_state_total_bytes() == (900 + 220) * MIB
    assert replace(sm, gdn_num_slots=5).gdn_state_total_bytes() == (500 + 220) * MIB
    assert sm.fixed_overhead_bytes() - _static().fixed_overhead_bytes() == 220 * MIB


def test_graph_capture_and_prefill_peaks_are_not_summed():
    p = _planner(calib=_calib(graph_capture_peak=900 * MIB))
    assert p.ledger(_Cfg, 4096, 0, 256)["transient"] == 900 * MIB
    assert p.ledger(_Cfg, 8192, 0, 256)["transient"] == 1700 * MIB


def test_solve_funds_context_then_largest_chunk_then_experts_within_budget():
    p = _planner()
    budget = 8 * 1024 * MIB
    chunk, experts, pages = p.phase_fg_solve_chunk_and_experts(_Cfg, budget)
    ledger = p.ledger(_Cfg, chunk, experts, pages)
    assert pages >= 16384 // 64
    assert chunk == 8192
    assert sum(ledger.values()) <= budget
    # residual smaller than one more expert slot: nothing left unspent at slot grain
    assert budget - sum(ledger.values()) < p.static_model.expert_bytes_per_slot


def test_solve_shrinks_chunk_before_failing_expert_floor():
    p = _planner()
    floor_at = lambda c: sum(p.ledger(_Cfg, c, 512, 256).values())  # noqa: E731
    budget = floor_at(4096) + MIB  # fits the floor at 4096, not at 8192
    assert floor_at(8192) > budget
    chunk, experts, _ = p.phase_fg_solve_chunk_and_experts(_Cfg, budget)
    assert chunk < 8192 and experts >= 512


def test_infeasible_reports_required_available_shortfall_and_owners():
    p = _planner()
    with pytest.raises(RuntimeError) as e:
        p.phase_fg_solve_chunk_and_experts(_Cfg, 1024 * MIB)
    msg = str(e.value)
    for word in ("required=", "available=", "shortfall=", "largest owners", "experts="):
        assert word in msg
    assert "o contexto pedido de" in msg and "o máximo possível é" in msg
    most = int(msg.split("o máximo possível é ")[1].split()[0])
    assert 0 <= most < _Cfg.max_seq_len and most % 1024 == 0


def test_kv_term_prices_dummy_page_and_fixed_tiers():
    p = _planner()
    assert p.ledger(_Cfg, 0, 0, 256)["kv"] == 257 * (MIB // 4) + MIB


def test_budget_owners_are_not_double_counted():
    """Weights, CUDA context and non-PyTorch growth are already outside the
    measured budget; the ledger must not charge them again."""
    p = _planner(calib=_calib(non_pytorch_growth=3 * 1024 * MIB))
    ledger = p.ledger(_Cfg, 4096, 0, 256)
    assert "weights" not in ledger and "non_pytorch" not in ledger
    assert sum(ledger.values()) < 3 * 1024 * MIB


def test_gdn_slots_budgeted_at_built_pool_size():
    class Group:
        pass

    class MC:
        def linear_attention_group(self):
            return Group()

    class C:
        max_running_req = 1
        cache_type = "hybrid_radix"
        linear_state_cache_ratio = 2.0
        model_config = MC()

    # 1 request: 4 working + ceil(2 * 1) snapshots + 1 padding
    assert _linear_pool_num_slots(C) == 7 > _linear_pool_min_slots(C)


def test_ram_tier_keeps_only_the_hot_floor_on_device():
    from types import SimpleNamespace

    from freetoken.engine.memory_planner import MemoryPlanner

    planner = MemoryPlanner.__new__(MemoryPlanner)
    planner.static_model = SimpleNamespace(kv_pages_for_context=lambda t: -(-t // 64))
    config = SimpleNamespace(max_seq_len=262144, kv_reserve_tokens=8192)
    assert planner._device_kv_pages(config) == 4096
    planner._host_pages = 4096  # whole context has a RAM page
    assert planner._device_kv_pages(config) == 128
    config.model_config = SimpleNamespace(kv_ram_tier_certified=True)
    config.kv_format = "fp8"
    assert planner._device_kv_pages(config) == 0
    planner._host_pages = 1024  # RAM covers part: the device holds the remainder
    assert planner._device_kv_pages(config) == 3072
    short = SimpleNamespace(max_seq_len=4096, kv_reserve_tokens=8192)
    assert planner._device_kv_pages(short) == 64
