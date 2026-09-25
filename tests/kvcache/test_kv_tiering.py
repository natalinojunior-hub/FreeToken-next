import pytest
import torch
from types import SimpleNamespace

from freetoken.kvcache.kv_tiering import (
    KVLayout,
    KVPage,
    KVPagePool,
    KVPageRecord,
    Residency,
    tiering_safe,
)
from freetoken.engine.config import EngineConfig
from freetoken.distributed import DistributedInfo


def page(i):
    return KVPage(i, i * 4, host=torch.zeros(8), device=torch.zeros(8))


def test_versioned_ownership_and_duplicate_release():
    pool = KVPagePool([page(0)], 1)
    p = pool.prefetch(0, "r1")
    generation = p.generation
    assert p.residency is Residency.RESIDENT
    assert not pool.evict(0, "r2", generation)
    assert pool.evict(0, "r1", generation)
    assert p.generation > generation


def test_pool_backpressure_and_cancel_cleanup():
    pool = KVPagePool([page(0), page(1)], 1)
    pool.prefetch(0, "r1")
    with pytest.raises(MemoryError):
        pool.admit(1, "r2")
    assert pool.cancel("r1")
    pool.prefetch(1, "r2")
    assert pool.table.get(1).residency is Residency.RESIDENT


def test_close_requires_restart_cleanup():
    pool = KVPagePool([page(0)], 1)
    pool.prefetch(0, "r")
    with pytest.raises(RuntimeError, match="owned pages"):
        pool.close()
    pool.evict(0, "r")
    pool.close()
    assert pool._free == []


def test_wrong_position_or_backing_is_rejected():
    p = page(0)
    p.device = torch.zeros(4)
    with pytest.raises(ValueError):
        KVPagePool([p], 1).prefetch(0, "r")


def test_capability_gate_keeps_fallback():
    assert tiering_safe(cuda_graph=True, stable_indirection=True, pinned_host=True)
    assert not tiering_safe(cuda_graph=False, stable_indirection=True, pinned_host=True)


def test_graph_replay_refuses_cold_pages_and_tracks_hits():
    pool = KVPagePool([page(0)], 1)
    assert not pool.can_replay([0])
    pool.prefetch(0, "r")
    assert pool.can_replay([0])
    assert pool.telemetry.cold_hits == 1
    assert pool.telemetry.rejected_replays == 1
    batch = type("Batch", (), {"kv_page_ids": [0]})()
    assert pool.can_replay_batch(batch)
    assert not pool.can_replay_batch(type("Batch", (), {})())


def test_eviction_round_trip_and_duplicate_claim():
    p = page(0)
    pool = KVPagePool([p], 1)
    pool.prefetch(0, "r")
    with pytest.raises(RuntimeError):
        pool.admit(0, "r")
    p.device.fill_(7)
    generation = p.generation
    assert pool.evict(0, "r", generation)
    assert torch.equal(p.host, torch.full_like(p.host, 7))
    pool.prefetch(0, "r2")
    assert torch.equal(p.device, p.host)
    assert not pool.evict(0, "r2", generation)


def test_stale_host_backing_is_rejected_after_first_load():
    p = page(0)
    pool = KVPagePool([p], 1, verify_checksums=True)
    pool.prefetch(0, "r")
    pool.evict(0, "r")
    p.host.add_(1)
    with pytest.raises(RuntimeError, match="stale host backing"):
        pool.prefetch(0, "r2")


def test_bad_backing_does_not_claim_page():
    p = page(0)
    p.device = torch.zeros(4)
    with pytest.raises(ValueError):
        KVPagePool([p], 1)
    assert p.owner is None
    assert p.device_slot is None


def test_runtime_rejects_unimplemented_auto_tiering():
    with pytest.raises(NotImplementedError, match="only 'off' and 'force'"):
        EngineConfig.__post_init__(SimpleNamespace(kv_tiering="auto", kv_ram_tokens=0))


def test_force_tiering_requires_kv_ram_tokens():
    with pytest.raises(ValueError, match="kv-ram-tokens"):
        EngineConfig.__post_init__(SimpleNamespace(kv_tiering="force", kv_ram_tokens=0))


def _config(**overrides):
    kwargs = dict(
        model_path="/tmp/freetoken-test-model",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.float16,
    )
    kwargs.update(overrides)
    return EngineConfig(**kwargs)


def test_off_tiering_ignores_kv_ram_tokens():
    config = _config(kv_tiering="off", kv_ram_tokens=4096)
    assert config.kv_tiering == "off"
    assert config.kv_ram_tokens == 4096  # stored but has no effect (off keeps the all-VRAM path)


def test_force_tiering_accepts_positive_kv_ram_tokens():
    config = _config(kv_tiering="force", kv_ram_tokens=4096)
    assert config.kv_ram_tokens == 4096


def test_canonical_page_record_validates_backend_layout():
    record = KVPageRecord(
        request_id=1,
        sequence_id=2,
        logical_position=64,
        generation=3,
        layout=KVLayout("qsa", (64, 8), (64, 8), (8, 1), (8, 1), 64),
        head_mapping=(0, 1),
        group_size=8,
        rope_position=64,
        qsa_group=8,
    )
    record.validate()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_round_trip_fences_eviction():
    p = KVPage.pinned((16,), torch.float32, 1, 0)
    pool = KVPagePool([p], 1, device="cuda")
    p.host.fill_(3)
    pool.prefetch(1, "r")
    torch.cuda.synchronize()
    assert torch.equal(p.device.cpu(), p.host)
    p.device.fill_(9)
    assert not pool.evict(1, "r")
    torch.cuda.synchronize()
    assert pool.evict(1, "r")
    assert torch.equal(p.host, torch.full_like(p.host, 9))
