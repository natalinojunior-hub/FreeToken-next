"""Canonical VRAM ledger arithmetic of MemoryPlanner (no GPU).

Covers: one ledger term per owner, transient = max(prefill, graph capture) not
their sum, context floor always funded, experts take the residual, exact
infeasibility report, and the two-point transient model.
"""

from __future__ import annotations

import pytest

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

    # 1 request: 4 working + max(4, 2) snapshot + 1 padding
    assert _linear_pool_num_slots(C) == 9 > _linear_pool_min_slots(C)


def test_ram_tier_keeps_only_the_hot_floor_on_device():
    from types import SimpleNamespace

    from freetoken.engine.memory_planner import MemoryPlanner

    planner = MemoryPlanner.__new__(MemoryPlanner)
    planner.static_model = SimpleNamespace(kv_pages_for_context=lambda t: -(-t // 64))
    config = SimpleNamespace(max_seq_len=262144, kv_reserve_tokens=8192)
    assert planner._device_kv_pages(config) == 4096
    planner._host_pages = 4096  # whole context has a RAM page
    assert planner._device_kv_pages(config) == 128
    planner._host_pages = 1024  # RAM covers part: the device holds the remainder
    assert planner._device_kv_pages(config) == 3072
    short = SimpleNamespace(max_seq_len=4096, kv_reserve_tokens=8192)
    assert planner._device_kv_pages(short) == 64


def test_auto_tiering_spills_when_the_all_vram_solve_fails(monkeypatch):
    """Phase C passes the all-VRAM floor, the real solve cannot fund it: spill, don't refuse."""
    from types import SimpleNamespace

    import freetoken.engine.memory_planner as mp

    planner = mp.MemoryPlanner.__new__(mp.MemoryPlanner)
    snap = mp.PhysicalMemorySnapshot(10 << 30, 16 << 30, 0, 0, 0, 0)
    budgets = []

    def solve(config, budget):
        budgets.append((budget, planner._host_pages))
        if not planner._host_pages:
            raise RuntimeError("o contexto pedido ... o máximo possível é ...")
        return 8192, 100, 128

    def baseline():
        planner.baseline_snapshot = snap

    def static_model(**kw):
        planner.static_model = SimpleNamespace(min_expert_slots=1, page_tokens=64)

    def mandatory():
        planner.post_mandatory_snapshot = snap

    for name, fn in {
        "phase_a_hardware_baseline": baseline,
        "phase_b_build_static_model": static_model,
        "phase_c_build_minimal_config": lambda c, m: None,
        "phase_a_post_mandatory": mandatory,
        "phase_d_runtime_calibration": lambda c, m: None,
        "_cleanup_probe_artifacts": lambda: None,
        "_device_kv_pages": lambda c: 1,
        "ledger": lambda *a: {"experts": 1, "kv": 1, "transient": 1},
        "phase_fg_solve_chunk_and_experts": solve,
        "phase_h_construct_final_pools": lambda *a: (None, None, None),
        "phase_i_final_validation": lambda *a: (True, ""),
    }.items():
        monkeypatch.setattr(planner, name, fn, raising=False)
    monkeypatch.setattr(mp, "take_physical_snapshot", lambda d: snap)
    monkeypatch.setattr(mp, "attach_offload_moe_cache", lambda *a: None)
    monkeypatch.setattr(mp.torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(mp.torch.cuda, "empty_cache", lambda: None)
    planner.device = None
    config = SimpleNamespace(max_seq_len=262144, kv_reserve_tokens=8192)

    plan = planner.plan(
        config, None, 256, 40, False, None, 1, 262144, 64,
        weights_bytes=1 << 30, host_reserve_bytes=1 << 20, host_pages=4096,
        spill_only_if_needed=True,
    )  # fmt: skip
    assert planner.host_pages == 4096 and plan.kv_pages == 128
    assert budgets == [(10 << 30, 0), ((10 << 30) - (1 << 20), 4096)]

    planner._host_pages = 0
    budgets.clear()
    try:
        planner.plan(config, None, 256, 40, False, None, 1, 262144, 64, weights_bytes=1 << 30)
    except RuntimeError:
        pass
    else:
        raise AssertionError("without a RAM tier the refusal must stand")
