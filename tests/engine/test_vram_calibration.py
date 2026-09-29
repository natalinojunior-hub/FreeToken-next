"""CPU regressions for allocator and VMM VRAM calibration."""

from types import SimpleNamespace

import torch

from freetoken.engine import engine as engine_module
from freetoken.engine.vram_ledger import Kind, open_ledger

_GIB = 1 << 30


class _Logger:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info_rank0(self, message):
        self.infos.append(message)

    def warning_rank0(self, message):
        self.warnings.append(message)


def _engine(monkeypatch, *, allocated, vmm_backed, expert_bytes):
    ledger = open_ledger(
        device_total_bytes=64 * _GIB,
        baseline_free=32 * _GIB,
        memory_ratio=1.0,
        weights_bytes=1 * _GIB,
    )
    ledger.charge("cache:expert", expert_bytes, Kind.PERSISTENT)
    ledger.charge("cache:kv", 1 * _GIB, Kind.PERSISTENT)
    ledger.log = lambda: None

    engine = engine_module.Engine.__new__(engine_module.Engine)
    engine.device = 0
    engine.config = SimpleNamespace()
    engine.vram_ledger = ledger
    engine.moe_offload_cache = SimpleNamespace(
        _vmm_arenas=([SimpleNamespace(backed_bytes=vmm_backed)] if vmm_backed else [])
    )
    engine._transient_probe_base = None
    engine._log_context_feasibility = lambda config, account: None
    logger = _Logger()
    monkeypatch.setattr(engine_module, "logger", logger)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: allocated)
    return engine, logger


def _budget_snapshot(ledger):
    return (
        ledger.reserve_bytes,
        ledger.ceiling_bytes,
        ledger.engine_overhead_bytes(),
        ledger.pool_budget_bytes(),
        ledger.held_bytes(),
        ledger.uncommitted_bytes(),
    )


def test_vmm_backing_reconciles_without_overmodel_warning_or_budget_change(monkeypatch):
    vmm_backed = 2 * _GIB
    engine, logger = _engine(
        monkeypatch,
        allocated=2 * _GIB,
        vmm_backed=vmm_backed,
        expert_bytes=vmm_backed,
    )
    before = _budget_snapshot(engine.vram_ledger)

    engine._calibrate_vram_ledger()

    assert engine.vram_ledger.bytes_of("measured:allocator-held") == 2 * _GIB
    assert engine.vram_ledger.bytes_of("measured:vmm-backed") == vmm_backed
    assert not logger.warnings
    assert not any("over-modelled" in message for message in logger.infos)
    assert _budget_snapshot(engine.vram_ledger) == before


def test_vmm_does_not_hide_a_genuine_under_account(monkeypatch):
    from freetoken.tuning import diagnostics

    monkeypatch.setattr(diagnostics, "log_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(diagnostics, "vram_snapshot", lambda device: {})
    engine, logger = _engine(
        monkeypatch,
        allocated=3 * _GIB,
        vmm_backed=1 * _GIB,
        expert_bytes=1 * _GIB,
    )

    engine._calibrate_vram_ledger()

    assert len(logger.warnings) == 1
    assert "under-modelled" in logger.warnings[0]
    assert "allocator and VMM hold" in logger.warnings[0]


def test_full_back_cache_is_not_counted_twice(monkeypatch):
    engine, logger = _engine(
        monkeypatch,
        allocated=4 * _GIB,
        vmm_backed=0,
        expert_bytes=2 * _GIB,
    )
    before = _budget_snapshot(engine.vram_ledger)

    engine._calibrate_vram_ledger()

    assert engine.vram_ledger.bytes_of("measured:allocator-held") == 4 * _GIB
    assert engine.vram_ledger.bytes_of("measured:vmm-backed") == 0
    assert _budget_snapshot(engine.vram_ledger) == before
    assert not any("under-modelled" in message for message in logger.warnings)
    assert not any("over-modelled" in message for message in logger.infos)
