import json
import time

import torch

from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.moe.trace import MoeTracer


def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FREETOKEN_MOE_TRACE", str(tmp_path))
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=4,
        cache_size=6,
        device=torch.device("cpu"),
    )
    cache.set_bank_sources(
        {
            "gate_up": [torch.zeros(4, 32, 8), torch.zeros(4, 32, 8)],
            "down": [torch.zeros(4, 8, 16), torch.zeros(4, 8, 16)],
        }
    )
    return cache


def test_snapshot_describes_real_shared_pools(tmp_path, monkeypatch):
    cache = _cache(tmp_path, monkeypatch)
    snapshot = json.loads((tmp_path / "snapshot.json").read_text())

    assert snapshot["cache_size_rows"] == 6
    assert snapshot["pool_caps"] == cache.pool_caps
    assert snapshot["pool_of_layer"] == cache.pool_of_layer
    assert len(snapshot["pools"]) == len(cache.pools)
    for actual, pool in zip(snapshot["pools"], cache.pools, strict=True):
        assert actual["layers"] == list(pool.layers)
        assert actual["row_bytes"] == list(pool.row_bytes)
    assert snapshot["layers"][0]["total_bytes"] == 32 * 8 * 4 + 8 * 16 * 4
    cache.tracer.close()


def test_trace_is_disabled_without_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("FREETOKEN_MOE_TRACE", raising=False)
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=4,
        cache_size=4,
        device=torch.device("cpu"),
    )
    assert cache.tracer is None


def test_trace_records_misses_timing_and_pool_residency(tmp_path, monkeypatch):
    cache = _cache(tmp_path, monkeypatch)
    cache.tracer.write_initial_residency([-1] * cache.cache_size, [0] * cache.cache_size)
    cache._trace_kind = "verify"
    cache._trace_ids = [1, 2, 3]
    cache._trace_pool_id = cache.pool_of_layer[0]
    cache._trace_evicted_ids = [1]
    cache._trace_resident_rows = 4
    cache.num_indices.fill_(2)

    cache._trace_copy_missing(0, None, time.perf_counter() - 0.001)
    cache.tracer.close()

    initial = json.loads((tmp_path / "initial_residency.json").read_text())
    assert initial["id_of_slot"] == [-1] * cache.cache_size
    rec = json.loads((tmp_path / "trace.jsonl").read_text())
    row_bytes = 32 * 8 * 4 + 8 * 16 * 4
    assert rec["access_step"] == 0
    assert rec["kind"] == "verify"
    assert rec["pool"] == cache.pool_of_layer[0]
    assert rec["expert_ids"] == [1, 2, 3]
    assert rec["missing"] == 2
    assert rec["miss_bytes"] == 2 * row_bytes
    assert rec["bank_bytes"] == {"gate_up": 32 * 8 * 4, "down": 8 * 16 * 4}
    assert rec["evicted_ids"] == [1]
    assert rec["resident_rows"] == 4
    assert rec["transfer_ms"] > 0
    assert rec["available_vram_bytes"] is None
