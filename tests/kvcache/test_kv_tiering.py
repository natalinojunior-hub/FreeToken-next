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


def test_config_accepts_auto_and_rejects_unknown_tiering():
    EngineConfig.__post_init__(
        SimpleNamespace(kv_tiering="auto", kv_ram_tokens=0, moe_backend=None)
    )
    with pytest.raises(ValueError, match="tiering"):
        EngineConfig.__post_init__(SimpleNamespace(kv_tiering="swap", kv_ram_tokens=0))


def test_kv_ram_budget_refuses_with_max_context(monkeypatch):
    from freetoken.engine import engine as eng

    gib = 1 << 30
    monkeypatch.setattr(eng, "_meminfo", lambda: {"MemTotal": 96 * gib, "MemAvailable": 16 * gib})
    per_token = {
        torch.bfloat16: 25_600,
        torch.float8_e4m3fn: 12_800,
        "turbo8": 13_000,
        "turbo4": 6_800,
    }
    pool = SimpleNamespace(
        host_tier_ram_bytes=lambda config, tokens, dtype=None: per_token.get(dtype, 5_200) * tokens
    )
    config = SimpleNamespace(max_seq_len=1 << 20, page_size=64, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="o contexto pedido de 1048576 tokens") as err:
        eng._check_kv_ram_budget(config, pool, (1 << 20) // 64)
    assert "o máximo possível é" in str(err.value)
    small = SimpleNamespace(
        max_seq_len=65536, page_size=64, dtype=torch.bfloat16, kv_ram_resolved_dtype=None
    )
    eng._check_kv_ram_budget(small, pool, 1024)
    # BF16 fits but FP8 dominates it (measured): auto starts at FP8; explicit bf16 still wins.
    assert small.kv_ram_resolved_dtype is torch.float8_e4m3fn
    small.kv_ram_dtype = "bf16"
    assert eng._kv_ram_dtype(small, pool, 1024) is torch.bfloat16
    # 256K: BF16 needs 6.25 GiB > 4.4 GiB budget, FP8 (3.1 GiB) fits.
    mid = SimpleNamespace(max_seq_len=1 << 18, page_size=64, dtype=torch.bfloat16)
    assert eng._kv_ram_dtype(mid, pool, (1 << 18) // 64) is torch.float8_e4m3fn
    # 512K: FP8 needs 6.25 GiB, turbo4 (3.3 GiB) fits; 768K: turbo4 5.0 GiB, turbo3 3.8 GiB.
    big = SimpleNamespace(max_seq_len=1 << 19, page_size=64, dtype=torch.bfloat16)
    assert eng._kv_ram_dtype(big, pool, (1 << 19) // 64) == "turbo4"
    assert eng._kv_ram_dtype(big, pool, 768 * 1024 // 64) == "turbo3"


def test_kv_ram_budget_narrows_format_after_model_load(monkeypatch):
    """The startup pick predates weights/expert banks in host RAM; the re-check narrows it."""
    from freetoken.engine import engine as eng

    gib = 1 << 30
    avail = {"MemTotal": 96 * gib, "MemAvailable": 24 * gib}
    monkeypatch.setattr(eng, "_meminfo", lambda: avail)
    per_token = {
        torch.bfloat16: 25_600,
        torch.float8_e4m3fn: 12_800,
        "turbo8": 13_000,
        "turbo4": 6_800,
    }
    pool = SimpleNamespace(
        host_tier_ram_bytes=lambda config, tokens, dtype=None: per_token.get(dtype, 5_200) * tokens
    )
    config = SimpleNamespace(max_seq_len=1 << 18, page_size=64, dtype=torch.bfloat16)
    config.kv_ram_resolved_dtype = eng._kv_ram_dtype(config, pool, (1 << 18) // 64)
    assert config.kv_ram_resolved_dtype is torch.float8_e4m3fn
    avail["MemAvailable"] = 14 * gib  # experts pinned: 10 GiB gone
    eng._check_kv_ram_budget(config, pool, (1 << 18) // 64)
    assert config.kv_ram_resolved_dtype == "turbo4"


def test_force_tiering_rejects_negative_kv_ram_tokens():
    with pytest.raises(ValueError, match="kv-ram-tokens"):
        EngineConfig.__post_init__(SimpleNamespace(kv_tiering="force", kv_ram_tokens=-1))
    # 0 sizes the RAM tier to the whole context.
    EngineConfig.__post_init__(
        SimpleNamespace(kv_tiering="force", kv_ram_tokens=0, moe_backend=None)
    )


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


def test_auto_tier_falls_back_to_vram_when_ram_is_short(monkeypatch):
    from freetoken.engine import engine as eng

    gib = 1 << 30
    monkeypatch.setattr(eng, "_meminfo", lambda: {"MemTotal": 96 * gib, "MemAvailable": 16 * gib})
    pool = SimpleNamespace(host_tier_ram_bytes=lambda config, tokens, dtype=None: 5_200 * tokens)
    config = SimpleNamespace(
        max_seq_len=1 << 20, page_size=64, dtype=torch.bfloat16, kv_ram_resolved_dtype="turbo3"
    )
    engine = SimpleNamespace(_pool_cls=pool, host_pages=(1 << 20) // 64, _host_reserve_bytes=1)
    engine._kv_tier_auto = True
    eng.Engine._fit_kv_ram_tier(engine, config)
    assert engine.host_pages == engine._host_reserve_bytes == 0
    assert config.kv_ram_resolved_dtype is None
    engine = SimpleNamespace(_pool_cls=pool, host_pages=(1 << 20) // 64, _kv_tier_auto=False)
    with pytest.raises(RuntimeError, match="o máximo possível é"):
        eng.Engine._fit_kv_ram_tier(engine, config)
    assert eng.EngineConfig.kv_tiering == "auto"


def test_auto_kv_ram_tier_requires_certified_family():
    """The auto default gates on a model-family capability, not a name match: only a certified
    family (qwen4) tiers KV in RAM by default; others keep all-VRAM until certified."""
    from freetoken.engine import engine as eng

    qsa_pool = type("QSAKVCache", (), {})
    device = SimpleNamespace(type="cuda")

    def cfg(certified):
        return SimpleNamespace(
            kv_format="auto",
            tp_info=SimpleNamespace(size=1),
            model_config=SimpleNamespace(kv_ram_tier_certified=certified),
        )

    assert eng._kv_ram_tier_unsupported(cfg(True), qsa_pool, device) is None
    reason = eng._kv_ram_tier_unsupported(cfg(False), qsa_pool, device)
    assert reason is not None and "certified" in reason

    class DensePool:
        pass

    assert eng._kv_ram_tier_unsupported(cfg(True), DensePool, device) == "DensePool KV pool"
