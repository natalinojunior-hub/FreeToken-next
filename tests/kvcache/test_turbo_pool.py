"""The compressed KV pool: what it stores, what it costs, and what it refuses.

The parity pins matter most: the engine sizes KV *before* the pool exists (kv_cost /
solve_num_pages) and reports what the pool actually holds afterwards (unit_bytes, and the VRAM
ledger's cache:kv line). If those three disagree, the plan promises memory the runtime did not
buy -- which is the exact class of bug the ledger was built to end.
"""

import pytest
import torch

from freetoken.kernel.triton import turbo_kv as tk
from freetoken.kvcache.turbo_pool import TurboMHAKVCache, packed_bytes_per_token

DEVICE = torch.device("cpu")
HEADS = 8
HEAD_DIM = 128
LAYERS = 4
PAGE = 1


def _pool(num_pages=9, book="turbo4", head_dim=HEAD_DIM, page_size=PAGE):
    return TurboMHAKVCache(
        num_kv_heads=HEADS,
        num_layers=LAYERS,
        head_dim=head_dim,
        num_pages=num_pages,
        page_size=page_size,
        dtype=torch.bfloat16,
        device=DEVICE,
        book=book,
    )


def test_store_then_read_back_matches_the_reference_codec():
    pool = _pool()
    rows = 9
    g = torch.Generator(device=DEVICE).manual_seed(3)
    k = torch.randn(rows, HEADS, HEAD_DIM, generator=g, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(rows, HEADS, HEAD_DIM, generator=g, device=DEVICE, dtype=torch.bfloat16)
    loc = torch.arange(rows, device=DEVICE)
    pool.store_kv(k, v, loc, layer_id=0)
    back = pool.decode_rows(0, "k")
    codes, norm = tk.quantize(k.reshape(-1, HEAD_DIM), "turbo4")
    want = tk.decode(codes, norm, "turbo4").reshape(rows, HEADS, HEAD_DIM)
    assert torch.equal(back, want.to(torch.bfloat16)), "the pool must not be a second codec"


def test_rows_land_in_the_named_slot_regardless_of_write_order():
    """out_loc is a token-slot index, not a batch position (pages are handed out non-contiguously
    and a CUDA-graph replay writes one scattered row at a time). Scattered stores must reproduce
    the batched store exactly."""
    g = torch.Generator(device=DEVICE).manual_seed(5)
    k = torch.randn(9, HEADS, HEAD_DIM, generator=g, device=DEVICE, dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    order = torch.tensor([4, 0, 8, 2, 7, 1, 5, 3, 6], device=DEVICE)
    scattered = _pool()
    for i, row in enumerate(order.tolist()):
        # int32 on purpose: the engine's page table is int32 while index_copy_ demands long, so a
        # store that only works with a long index is a startup crash, not a test artifact.
        scattered.store_kv(k[i : i + 1], v[i : i + 1], order[i : i + 1].to(torch.int32), layer_id=1)
    batched = _pool()
    batched.store_kv(k, v, torch.arange(9, device=DEVICE), layer_id=1)
    assert torch.equal(scattered._k_codes[1][order], batched._k_codes[1])
    assert torch.equal(scattered._k_norm[1][order], batched._k_norm[1])


def test_two_groups_per_row_pack_and_read_back():
    """head_dim 256 is two rotation groups in one row -- the path that is easy to get wrong when
    the norm is deduped per group."""
    pool = TurboMHAKVCache(4, 2, 256, 5, PAGE, torch.bfloat16, DEVICE, book="turbo4")
    g = torch.Generator(device=DEVICE).manual_seed(7)
    k = torch.randn(5, 4, 256, generator=g, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(5, 4, 256, generator=g, device=DEVICE, dtype=torch.bfloat16)
    pool.store_kv(k, v, torch.arange(5, device=DEVICE), layer_id=0)
    codes, norm = tk.quantize(k.reshape(-1, 256), "turbo4")
    want = tk.decode(codes, norm, "turbo4").reshape(5, 4, 256)
    assert torch.equal(pool.decode_rows(0, "k"), want.to(torch.bfloat16))
    assert pool._k_codes.shape == (2, 5, 4, 2 * tk.CODE_BYTES["turbo4"])


def test_unwritten_rows_stay_zero_and_a_bf16_reader_is_refused():
    pool = _pool()
    loc = torch.tensor([2, 3], device=DEVICE)
    pool.store_kv(
        torch.ones(2, HEADS, HEAD_DIM, device=DEVICE),
        torch.ones(2, HEADS, HEAD_DIM, device=DEVICE),
        loc,
        0,
    )
    assert pool._k_codes[0][0].eq(0).all() and pool._k_norm[0][0].eq(0).all()
    with pytest.raises(NotImplementedError, match="k_slab"):
        pool.k_cache(0)
    with pytest.raises(NotImplementedError, match="v_slab"):
        pool.v_cache(0)
    codes, norm = pool.k_slab(0)
    assert codes.dtype is torch.uint8 and norm.dtype is torch.float16


@pytest.mark.parametrize(("book", "row_bytes"), [("turbo3", 50), ("turbo4", 66)])
def test_packed_row_is_the_reference_layout_minus_the_deduped_norm(book, row_bytes):
    """The reference stores 14 B per 32-element turbo3 block (four identical norms per group) and
    66 B per 128-element turbo4 block. We keep the payload and store one norm per group."""
    ref_row = 56 if book == "turbo3" else 66
    assert tk.CODE_BYTES[book] + 2 == row_bytes
    assert row_bytes <= ref_row
    assert tk.BPV[book] == pytest.approx(row_bytes * 8 / tk.QK_TURBO)


@pytest.mark.parametrize("book", ["turbo3", "turbo4"])
def test_unit_bytes_and_total_bytes_agree_with_the_allocation(book):
    pool = _pool(book=book)
    assert pool.unit_bytes()[0] * pool._tokens == pool.total_bytes()
    per_token = packed_bytes_per_token(HEAD_DIM, HEADS, book) * LAYERS
    assert pool.unit_bytes()[0] == per_token


def test_compression_ratio_against_bf16_is_the_number_the_plan_uses():
    """turbo4 must be ~3.9x smaller and turbo3 ~5.1x, per token per layer, both slabs included."""
    bf16 = 2 * HEADS * HEAD_DIM * 2 * LAYERS
    r4 = bf16 / _pool(book="turbo4").unit_bytes()[0]
    r3 = bf16 / _pool(book="turbo3").unit_bytes()[0]
    assert 3.8 < r4 < 4.0, f"turbo4 ratio {r4:.2f}"
    assert 5.0 < r3 < 5.3, f"turbo3 ratio {r3:.2f}"


def test_rebuild_resizes_in_place_and_keeps_object_identity():
    pool = _pool(num_pages=9)
    before = id(pool)
    old_bytes = pool.total_bytes()
    pool.rebuild_from_config(_FakeConfig(page_size=PAGE), num_pages=17)
    assert id(pool) == before
    assert pool.total_bytes() > old_bytes
    assert pool.unit_bytes()[0] * pool._tokens == pool.total_bytes()


def test_kv_cost_prices_the_same_bytes_the_pool_allocates():
    """The plan is written before the pool exists, so kv_cost and unit_bytes must be the same
    arithmetic. This is the invariant the ledger's cache:kv line is checked against."""
    from types import SimpleNamespace

    from freetoken.attention import AttnType
    from freetoken.models.config import KVCacheGroupSpec

    full = KVCacheGroupSpec(
        name="full",
        layer_ids=(0, 1, 2, 3),
        num_kv_heads=HEADS,
        head_dim=HEAD_DIM,
        sliding_window=None,
        mla=False,
        index_head_dim=0,
        num_index_layers=0,
        attn_type=AttnType.FULL,
    )
    swa = KVCacheGroupSpec(
        name="swa",
        layer_ids=(0, 1),
        num_kv_heads=2,
        head_dim=64,
        sliding_window=128,
        mla=False,
        index_head_dim=0,
        num_index_layers=0,
        attn_type=AttnType.SWA,
    )
    config = SimpleNamespace(
        page_size=PAGE,
        kv_format="turbo4",
        tp_info=SimpleNamespace(size=1),
        model_config=SimpleNamespace(kv_cache_group_specs=lambda: (full, swa)),
    )
    per_page, fixed, page_tokens, min_reserve = TurboMHAKVCache.kv_cost(config)
    assert page_tokens == PAGE and fixed == 0 and min_reserve == 0
    one_token = packed_bytes_per_token(HEAD_DIM, HEADS, "turbo4") * LAYERS
    assert per_page == one_token, "the SWA group must not be priced by this pool"
    pool = _pool(num_pages=2)
    assert pool.total_bytes() == 2 * one_token


class _FakeConfig:
    def __init__(self, page_size):
        self.page_size = page_size
        self.kv_format = "turbo4"


@pytest.mark.parametrize(
    ("head_dim", "message"),
    [(64, "multiple of 128"), (1024, "head_dim <= 512")],
)
def test_geometry_outside_the_supported_band_refuses(head_dim, message):
    with pytest.raises(ValueError, match=message):
        _pool(head_dim=head_dim)


def test_unknown_book_and_bad_layer_ids_refuse():
    # turbo8 is RAM-tier only (qsa/tiered.py); the device-compressed pool still only knows
    # turbo3/turbo4, so any other name -- including turbo8 -- must still refuse here.
    with pytest.raises(ValueError, match="unknown turbo book"):
        _pool(book="turbo9")
    with pytest.raises(ValueError, match="outside"):
        TurboMHAKVCache(HEADS, 4, HEAD_DIM, 8, PAGE, torch.bfloat16, DEVICE, layer_ids=[0, 9])


def test_layer_ids_remap_storage_slabs():
    """A hybrid-linear model has paged KV in a subset of layers; the pool must allocate the
    subset and route global ids, exactly like MHAKVCache does."""
    pool = TurboMHAKVCache(HEADS, 4, HEAD_DIM, 8, PAGE, torch.bfloat16, DEVICE, layer_ids=[1, 3])
    assert pool._k_codes.shape[0] == 2
    assert pool._dense(3) == 1
    with pytest.raises(KeyError):
        pool._dense(0)
