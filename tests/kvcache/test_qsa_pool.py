"""QSAKVCache tiers (paged K/V, compressed index slab, pending ring, scratch).

Pins the three things the QSA kernels and the startup budget both depend on: the compressed
slab is a 1/index_ratio shadow of the K/V pages with the scratch rows behind it, the ring and
scratch are fixed (concurrency-sized) and priced apart from the per-token slider, and the K/V
slabs cover the sparse layers only. The PLE conv history rides the GDN slots, so it must
follow every slot operation and show up in the state-pool byte account.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention import AttnType
from freetoken.kvcache.base import spec_kv_bytes_per_token
from freetoken.kvcache.qsa_pool import QSAKVCache
from freetoken.models.config import KVCacheGroupSpec

DEV = torch.device("cpu")

# Qwen3.8-Flash-Next: 48 layers, every 4th is QSA; 2 kv heads x 256, indexer 128 wide, ratio 4.
FULL_LAYER_IDS = tuple(range(3, 48, 4))
REAL_KV_BYTES = 2 * 256 * 2 * 2 * 12
REAL_INDEX_BYTES = 128 * 12 * 2 // 4


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.mha_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def _pool(num_pages=4, page_size=64, index_ratio=4, num_req_slots=4, ring_capacity=None):
    return QSAKVCache(
        num_kv_heads=2,
        num_layers=8,
        head_dim=64,
        num_pages=num_pages,
        page_size=page_size,
        dtype=torch.bfloat16,
        device=DEV,
        index_head_dim=32,
        num_index_layers=4,
        index_ratio=index_ratio,
        num_req_slots=num_req_slots,
        ring_capacity=ring_capacity,
        layer_ids=(1, 3, 5, 7),
    )


def _gpu_mtp_pool(*, kv_format="bf16", host_pages=0, host_dtype=None, preserve=True):
    return QSAKVCache(
        num_kv_heads=2,
        num_layers=3,
        head_dim=128,
        num_pages=4,
        page_size=64,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        index_head_dim=32,
        num_index_layers=3,
        index_ratio=4,
        num_req_slots=4,
        layer_ids=(0, 1, 2),
        kv_format=kv_format,
        mtp_layer_id=2,
        host_pages=host_pages,
        host_dtype=host_dtype,
        preserve_mtp_prefix=preserve,
    )


def _spec(
    *,
    index_ratio=4,
    attn_type=AttnType.QSA,
    num_kv_heads=2,
    head_dim=64,
    index_head_dim=32,
    num_index_layers=4,
    layer_ids=(1, 3, 5, 7),
):
    return KVCacheGroupSpec(
        name="full",
        layer_ids=layer_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sliding_window=None,
        index_head_dim=index_head_dim,
        num_index_layers=num_index_layers,
        index_ratio=index_ratio,
        attn_type=attn_type,
    )


def _config(spec, *, page_size=64, max_running_req=3):
    mc = SimpleNamespace(
        num_layers=8, has_swa_attention=False, has_linear_attention=True, model_is_mrope=False
    )
    mc.kv_cache_group_specs = lambda: (spec,)
    return SimpleNamespace(
        model_config=mc,
        page_size=page_size,
        dtype=torch.bfloat16,
        tp_info=SimpleNamespace(size=1),
        max_running_req=max_running_req,
    )


# --------------------------------------------------------------------- slab / ring geometry


def test_slab_ring_and_scratch_shapes():
    pool = _pool(num_pages=4)
    # 4 pages x 64 tokens / ratio 4 = 64 shadow rows, then one scratch row per request slot
    assert pool.cmp_scratch_base == 64
    assert pool.cmp_k_cache(0).shape == (64 + 4, 32)
    assert pool.cmp_k_cache(3).shape == (64 + 4, 32)
    assert pool.pending_ring(0).shape == (4, QSAKVCache.ring_capacity_for(4), 32)
    assert pool.cmp_k_cache(0).abs().sum().item() == 0.0
    assert pool.k_cache(1).shape == (4, 64, 2, 64)


def test_kv_slabs_cover_sparse_layers_only():
    # Copying the BSA branch (no layer_ids) would back all 8 model layers instead of 4.
    pool = _pool()
    assert pool._kv_buffer.shape[1] == 4
    pool.k_cache(7)
    with pytest.raises(KeyError):
        pool.k_cache(0)


def test_ring_capacity_and_ratio_are_parameters():
    pool = _pool(num_pages=8, index_ratio=2, num_req_slots=3, ring_capacity=6)
    assert pool.index_ratio == 2 and pool.ring_capacity == 6
    assert pool.cmp_scratch_base == 8 * 64 // 2
    assert pool.pending_ring(0).shape == (3, 6, 32)


def test_ring_capacity_formula_and_floor():
    assert QSAKVCache.ring_capacity_for(4) == 4
    assert QSAKVCache.ring_capacity_for(8) == 8
    assert QSAKVCache.ring_capacity_for(4, num_speculative_tokens=3) == 8
    with pytest.raises(ValueError, match="ring_capacity"):
        _pool(ring_capacity=2, index_ratio=4)


def test_group_must_not_straddle_a_page():
    with pytest.raises(ValueError, match="divisible"):
        _pool(page_size=6, index_ratio=4)


def test_index_slab_needs_a_two_byte_dtype():
    with pytest.raises(AssertionError, match="2 bytes"):
        QSAKVCache(
            num_kv_heads=2,
            num_layers=8,
            head_dim=64,
            num_pages=4,
            page_size=64,
            dtype=torch.float32,
            device=DEV,
            index_head_dim=32,
            num_index_layers=4,
            index_ratio=4,
            num_req_slots=4,
            layer_ids=(1, 3, 5, 7),
        )


def test_shadow_row_is_shared_by_a_whole_group():
    pool = _pool()
    cmp = pool.cmp_k_cache(2)
    row = torch.randn(32, dtype=torch.bfloat16)
    for slot in (64, 65, 66, 67):
        assert slot // pool.index_ratio == 16
    cmp[16] = row
    assert torch.equal(pool.cmp_k_cache(2)[16], row)
    # the other sparse layers keep their own rows
    assert pool.cmp_k_cache(1)[16].abs().sum().item() == 0.0


def test_rebuild_resizes_every_tier_and_keeps_identity():
    pool = _pool(num_pages=4)
    ident = id(pool)
    pool.rebuild(16)
    assert id(pool) == ident
    assert pool.k_cache(1).shape == (16, 64, 2, 64)
    assert pool.cmp_scratch_base == 16 * 64 // 4
    assert pool.cmp_k_cache(0).shape == (16 * 64 // 4 + 4, 32)
    assert pool.pending_ring(3).shape == (4, pool.ring_capacity, 32)
    assert pool._kv_buffer.shape[1] == 4  # sparse-layer slabs survive the resize
    pool.k_cache(7)


# ------------------------------------------------------------------------------ budgeting


def test_spec_bytes_per_token_divides_the_index_slab():
    spec = _spec(
        num_kv_heads=2,
        head_dim=256,
        index_head_dim=128,
        num_index_layers=12,
        layer_ids=FULL_LAYER_IDS,
    )
    config = _config(spec)
    assert spec_kv_bytes_per_token(spec, config) == REAL_KV_BYTES + REAL_INDEX_BYTES
    assert spec_kv_bytes_per_token(spec, config) == 24576 + 768

    # BSA/DSA keep one index row per token (ratio 1)
    bsa = _spec(
        num_kv_heads=2,
        head_dim=256,
        index_head_dim=128,
        num_index_layers=12,
        layer_ids=FULL_LAYER_IDS,
        index_ratio=1,
        attn_type=AttnType.BSA,
    )
    assert spec_kv_bytes_per_token(bsa, config) == REAL_KV_BYTES + 128 * 12 * 2


def test_kv_cost_prices_ring_and_scratch_as_fixed():
    spec = _spec()
    config = _config(spec, max_running_req=3)
    per_page, fixed, page_tokens, min_reserve = QSAKVCache.kv_cost(config)
    assert per_page == spec_kv_bytes_per_token(spec, config) * 64
    assert page_tokens == 64 and min_reserve == 0
    row = 32 * 4 * 2
    assert fixed == 4 * row * (QSAKVCache.ring_capacity_for(4) + 1)


def test_kv_cost_widens_the_ring_for_spec_mtp():
    spec = _spec()
    config = _config(spec, max_running_req=3)
    config.spec_mtp = 2
    _, fixed_spec, _, _ = QSAKVCache.kv_cost(config)
    row = 32 * 4 * 2
    assert fixed_spec == 4 * row * (QSAKVCache.ring_capacity_for(4, 2) + 1)
    assert QSAKVCache.ring_capacity_for(4, 2) > QSAKVCache.ring_capacity_for(4, 0)


def test_host_tier_device_bytes_prices_index_rope_and_staging():
    spec = _spec()
    config = _config(spec)
    config.model_config.model_is_mrope = True
    host_tokens = 128
    index_bytes = spec.index_head_dim * spec.num_index_layers * 2 // spec.index_ratio
    rope_bytes = 12  # _ROPE_POS_BYTES = 3 * 4
    staging_bytes = 2 * spec.num_kv_heads * spec.head_dim * config.dtype.itemsize
    expected = host_tokens * (index_bytes + rope_bytes + staging_bytes)
    assert QSAKVCache.host_tier_device_bytes(config, host_tokens) == expected


def test_host_tier_device_bytes_is_zero_without_host_tokens():
    spec = _spec()
    config = _config(spec)
    assert QSAKVCache.host_tier_device_bytes(config, 0) == 0


def test_unit_bytes_matches_the_cost_model():
    spec = _spec()
    config = _config(spec)
    pool = _pool()
    kv_bytes, swa_bytes = pool.unit_bytes()
    assert swa_bytes == 0
    # the scratch rows and the ring must NOT inflate the per-token slider
    assert kv_bytes == spec_kv_bytes_per_token(spec, config)
    assert kv_bytes * 64 == QSAKVCache.kv_cost(config)[0]


def test_resolve_pool_class_and_factory():
    from freetoken.kvcache import create_kvcache_pool, resolve_pool_class

    spec = _spec()
    mc = SimpleNamespace(
        model_is_mrope=False,
        num_layers=8,
        has_swa_attention=False,
        has_linear_attention=True,
        num_kv_heads=2,
        head_dim=64,
        dsv4_args=None,
    )
    mc.kv_cache_group_specs = lambda: (spec,)
    assert resolve_pool_class(mc) is QSAKVCache

    pool = create_kvcache_pool(
        mc, num_pages=4, page_size=64, dtype=torch.bfloat16, device=DEV, num_req_slots=4
    )
    assert isinstance(pool, QSAKVCache)
    assert pool._kv_buffer.shape[1] == 4  # not the model's 8 layers
    assert pool.cmp_k_cache(0).shape == (4 * 64 // 4 + 4, 32)

    with pytest.raises(ValueError, match="num_req_slots"):
        create_kvcache_pool(mc, num_pages=4, page_size=64, dtype=torch.bfloat16, device=DEV)


def test_create_kv_pool_fails_closed_for_non_qsa_host_tier():
    from freetoken.kvcache import create_kv_pool

    # No kv_cache_group_specs / dsv4_args -> resolve_pool_class falls back to MHAKVCache.
    mc = SimpleNamespace(dsv4_args=None, has_swa_attention=False, has_linear_attention=False)
    config = SimpleNamespace(model_config=mc, kv_format="auto")
    with pytest.raises(NotImplementedError, match="QSA BF16 KV pool"):
        create_kv_pool(config, num_pages=4, device=DEV, dtype=torch.bfloat16, host_pages=2)


def test_free_req_clears_pending_ring_and_scratch():
    pool = _pool(num_pages=4, num_req_slots=4)
    # Dirty slot 1 with non-zero values
    pool._pending_ring[1].fill_(42.0)
    pool._cmp_k_buffer[:, pool.cmp_scratch_base + 1].fill_(7.0)
    assert pool._pending_ring[1].abs().sum().item() > 0
    assert pool._cmp_k_buffer[:, pool.cmp_scratch_base + 1].abs().sum().item() > 0

    pool.free_req(1)
    assert pool._pending_ring[1].abs().sum().item() == 0.0
    assert pool._cmp_k_buffer[:, pool.cmp_scratch_base + 1].abs().sum().item() == 0.0


def test_free_req_clears_mtp_draft_slot():
    pool = QSAKVCache(
        num_kv_heads=2,
        num_layers=9,
        head_dim=64,
        num_pages=4,
        page_size=64,
        dtype=torch.bfloat16,
        device=DEV,
        index_head_dim=32,
        num_index_layers=5,
        index_ratio=4,
        num_req_slots=4,
        layer_ids=(1, 3, 5, 7, 8),
        mtp_layer_id=8,
    )
    assert pool._mtp_slot == 4

    # Dirty MTP slot across tiers
    pool._cmp_k_buffer[4].fill_(11.0)
    pool._pending_ring[:, 4].fill_(13.0)
    if hasattr(pool._pool, "_kv_buffer") and pool._pool._kv_buffer is not None:
        pool._pool._kv_buffer[:, 4].fill_(17.0)
    assert pool._cmp_k_buffer[4].abs().sum().item() > 0
    assert pool._pending_ring[:, 4].abs().sum().item() > 0

    pool.free_req(0)
    assert pool._cmp_k_buffer[4].abs().sum().item() == 0.0
    assert pool._pending_ring[:, 4].abs().sum().item() == 0.0
    if hasattr(pool._pool, "_kv_buffer") and pool._pool._kv_buffer is not None:
        assert pool._pool._kv_buffer[:, 4].abs().sum().item() == 0.0

    # Also verify Turbo4 variant
    pool_t4 = QSAKVCache(
        num_kv_heads=2,
        num_layers=9,
        head_dim=128,
        num_pages=4,
        page_size=64,
        dtype=torch.bfloat16,
        device=DEV,
        index_head_dim=32,
        num_index_layers=5,
        index_ratio=4,
        num_req_slots=4,
        layer_ids=(1, 3, 5, 7, 8),
        mtp_layer_id=8,
        kv_format="turbo4",
    )
    assert pool_t4._mtp_slot == 4
    pool_t4._pool._k_codes[4].fill_(5)
    pool_t4._pool._k_norm[4].fill_(1.5)
    assert pool_t4._pool._k_codes[4].abs().sum().item() > 0
    assert pool_t4._pool._k_norm[4].abs().sum().item() > 0
    pool_t4.free_req(0)
    assert pool_t4._pool._k_codes[4].abs().sum().item() == 0
    assert pool_t4._pool._k_norm[4].abs().sum().item() == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_free_req_preserves_mtp_prefix_and_other_request_state():
    pool = _gpu_mtp_pool()
    assert pool.preserve_mtp_prefix
    pool._kv_buffer.fill_(3.0)
    pool._cmp_k_buffer.fill_(5.0)
    pool._pending_ring.fill_(7.0)
    pool._cmp_k_buffer[:, pool.cmp_scratch_base + 1].fill_(11.0)
    pool._cmp_k_buffer[:, pool.cmp_scratch_base + 2].fill_(13.0)

    prefix_k = pool.k_cache(2)[:, :8].clone()
    prefix_v = pool.v_cache(2)[:, :8].clone()
    prefix_cmp = pool._cmp_k_buffer[:, :8].clone()
    other_k = pool.k_cache(0)[:, 8:16].clone()
    other_v = pool.v_cache(0)[:, 8:16].clone()
    other_ring = pool._pending_ring[2].clone()
    other_scratch = pool._cmp_k_buffer[:, pool.cmp_scratch_base + 2].clone()

    pool.free_req(1)

    assert torch.equal(pool.k_cache(2)[:, :8], prefix_k)
    assert torch.equal(pool.v_cache(2)[:, :8], prefix_v)
    assert torch.equal(pool._cmp_k_buffer[:, :8], prefix_cmp)
    assert torch.equal(pool.k_cache(0)[:, 8:16], other_k)
    assert torch.equal(pool.v_cache(0)[:, 8:16], other_v)
    assert torch.equal(pool._pending_ring[2], other_ring)
    assert torch.equal(pool._cmp_k_buffer[:, pool.cmp_scratch_base + 2], other_scratch)
    assert torch.count_nonzero(pool._pending_ring[1]) == 0
    assert torch.count_nonzero(pool._cmp_k_buffer[:, pool.cmp_scratch_base + 1]) == 0


@pytest.mark.parametrize("kv_format", ("bf16", "turbo4"))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_zero_mtp_first_token_changes_only_target_layer_row(kv_format):
    pool = _gpu_mtp_pool(kv_format=kv_format)
    logical = pool._page_size + 5
    out_loc = torch.tensor([logical], dtype=torch.int32, device=pool.device)

    if kv_format == "bf16":
        pool._kv_buffer.fill_(9.0)
        k_before, v_before = pool.k_cache(2).clone(), pool.v_cache(2).clone()
        target_before = [
            (pool.k_cache(layer).clone(), pool.v_cache(layer).clone()) for layer in (0, 1)
        ]
        pool.zero_mtp_first_token(out_loc)
        expected_k, expected_v = k_before.clone(), v_before.clone()
        expected_k[1, 5].zero_()
        expected_v[1, 5].zero_()
        assert torch.equal(pool.k_cache(2), expected_k)
        assert torch.equal(pool.v_cache(2), expected_v)
        for layer, (k, v) in zip((0, 1), target_before):
            assert torch.equal(pool.k_cache(layer), k)
            assert torch.equal(pool.v_cache(layer), v)
        return

    pool._pool._k_codes.fill_(7)
    pool._pool._v_codes.fill_(9)
    pool._pool._k_norm.fill_(2.0)
    pool._pool._v_norm.fill_(3.0)
    before = [
        t.clone()
        for t in (
            pool._pool._k_codes,
            pool._pool._k_norm,
            pool._pool._v_codes,
            pool._pool._v_norm,
        )
    ]
    pool.zero_mtp_first_token(out_loc)
    from freetoken.kernel.triton.turbo_kv import quantize

    zeros = torch.zeros(2, 128, device=pool.device, dtype=torch.bfloat16)
    zero_codes, zero_norms = quantize(zeros, "turbo4")
    expected = [t.clone() for t in before]
    dense = 2
    expected[0][dense, logical] = zero_codes.reshape_as(expected[0][dense, logical])
    expected[1][dense, logical] = zero_norms.reshape_as(expected[1][dense, logical])
    expected[2][dense, logical] = zero_codes.reshape_as(expected[2][dense, logical])
    expected[3][dense, logical] = zero_norms.reshape_as(expected[3][dense, logical])
    for actual, wanted in zip(
        (pool._pool._k_codes, pool._pool._k_norm, pool._pool._v_codes, pool._pool._v_norm),
        expected,
    ):
        assert torch.equal(actual, wanted)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_zero_mtp_first_token_uses_page_map_for_fp8_host_tier():
    pool = _gpu_mtp_pool(
        host_pages=2,
        host_dtype=torch.float8_e4m3fn,
    )
    # Route logical page zero to physical RAM page three; zeroing must honor this indirection.
    pool.page_map[0] = 3
    logical = 5
    out_loc = torch.tensor([logical], dtype=torch.int32, device=pool.device)
    pool._kv_buffer.fill_(6.0)
    for layer in (0, 1, 2):
        host_k, host_v = pool.host_kv(layer)
        host_k.fill_(4.0)
        host_v.fill_(5.0)
    host_before = [tuple(t.clone() for t in pool.host_kv(layer)) for layer in (0, 1, 2)]
    device_before = pool._kv_buffer.clone()

    pool.zero_mtp_first_token(out_loc)

    for layer, (k_before, v_before) in zip((0, 1), host_before[:2]):
        host_k, host_v = pool.host_kv(layer)
        assert torch.equal(host_k, k_before)
        assert torch.equal(host_v, v_before)
    target_k, target_v = pool.host_kv(2)
    expected_k, expected_v = host_before[2][0].clone(), host_before[2][1].clone()
    expected_k[1, 5].zero_()
    expected_v[1, 5].zero_()
    assert torch.equal(target_k, expected_k)
    assert torch.equal(target_v, expected_v)
    assert torch.equal(pool._kv_buffer, device_before)
