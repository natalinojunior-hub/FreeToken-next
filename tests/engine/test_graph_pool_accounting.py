"""Graph ownership replaces startup estimates with physically retained bytes."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.engine import Engine
from freetoken.engine.vram_ledger import Kind, VramLedger


@pytest.mark.parametrize(
    "retained,walked,expected", [(24, 0, 24), (24, 3, 24), (0, 3, 3), (-1, 3, 3)]
)
def test_graph_charge_replaces_placeholder_and_includes_private_pool(
    monkeypatch, retained, walked, expected
):
    mib = 1 << 20
    charges = {"graph:pool": (200 * mib, Kind.SEMI_PERSISTENT, "startup estimate")}
    engine = Engine.__new__(Engine)
    engine.graph_runner = SimpleNamespace(graph_bs_list=[1])
    engine.vram_ledger = SimpleNamespace(
        charge=lambda name, size, kind, detail: charges.update({name: (size, kind, detail)})
    )
    monkeypatch.setattr("freetoken.engine.engine.tensor_bytes", lambda _runner: walked * mib)
    engine._charge_graph_pool(retained * mib)
    size, kind, detail = charges["graph:pool"]
    assert size == expected * mib
    assert kind is Kind.SEMI_PERSISTENT
    assert "retained allocator" in detail


@pytest.mark.parametrize("graph_peak,expected", [(0, 800), (900, 900)])
def test_prefill_calibration_replaces_overlapping_estimates_and_credits_slack_once(
    monkeypatch, graph_peak, expected
):
    mib = 1 << 20
    engine = Engine.__new__(Engine)
    engine.device = torch.device("cuda")
    engine.moe_offload_cache = None
    ledger = engine.vram_ledger = VramLedger(16000 * mib, 15000 * mib, 0.98)
    replaced = (
        "transient:autotune",
        "transient:activations",
        "transient:gdn-prefill",
        "graph:capture-peak",
        "reserve:fragmentation",
    )
    for name in replaced:
        ledger.charge(name, 1000 * mib, Kind.TRANSIENT)
    preserved = {
        "transient:mm-encoder": (192 * mib, Kind.TRANSIENT),
        "workspace:attention": (128 * mib, Kind.SEMI_PERSISTENT),
        "graph:pool": (24 * mib, Kind.SEMI_PERSISTENT),
    }
    for name, (size, kind) in preserved.items():
        ledger.charge(name, size, kind)
    chunks = []
    calibration = SimpleNamespace(
        transient_at=lambda chunk: (chunks.append(chunk), 700 * mib)[1],
        baseline_allocator_slack=100 * mib,
        graph_capture_peak=graph_peak * mib,
    )
    engine._adopt_prefill_calibration(calibration, 8192)
    assert chunks == [8192]
    assert engine._prefill_transient_reserve == expected * mib
    assert ledger.bytes_of("transient:prefill-calibrated") == expected * mib
    assert all(name not in ledger.charges for name in replaced)
    assert all(
        ledger.charges[name].nbytes == size and ledger.charges[name].kind == kind
        for name, (size, kind) in preserved.items()
    )
    assert ledger.reserve_bytes == (expected + 192) * mib
    monkeypatch.setattr("freetoken.engine.engine._vmm_resize_slack", lambda _cache: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 300 * mib)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 200 * mib)
    # Exactly one live-slack credit makes this boundary fit. One byte less remains pressure.
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda _device: ((expected - 100) * mib, 16000 * mib)
    )
    assert engine.guard_vram_before_forward(prefill=True) is False
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda _device: ((expected - 100) * mib - 1, 16000 * mib)
    )
    assert engine.guard_vram_before_forward(prefill=True) is True
