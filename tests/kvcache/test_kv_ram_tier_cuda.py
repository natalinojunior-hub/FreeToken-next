"""CUDA checks for the KV RAM tier: tiered store/gather/zero kernels, host-tier QSA attention,
and the MHAKVCache/QSAKVCache host_pages wiring around them.
"""

import pytest
import torch

from freetoken.kernel.triton.qsa.attend import qsa_sparse_paged_attention
from freetoken.kernel.triton.qsa.tiered import gather_pages, swap_pages, tiered_store_kv, zero_tier
from freetoken.kvcache.mha_pool import registered_host_empty
from freetoken.kvcache.qsa_pool import QSAKVCache
from freetoken.kvcache.mha_pool import MHAKVCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = torch.device("cuda")
DTYPE = torch.bfloat16


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.mha_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def host_empty(shape):
    return registered_host_empty(shape, DTYPE)


def reference_attention(q, k_cache, v_cache, host_kv, logical_indices, block_table, token_to_req):
    """Non-flash softmax recompute of the same math the kernel does, for correctness checks."""
    rows, hq, d = q.shape
    kvh = k_cache.shape[2]
    group = hq // kvh
    page_size = k_cache.shape[1]
    num_dev = k_cache.shape[0]
    ptw = block_table.shape[1]
    num_host = host_kv[0].shape[0] if host_kv is not None else 0

    tok = logical_indices.long()
    safe_tok = tok.clamp(min=0)
    page = safe_tok // page_size
    off = safe_tok % page_size
    valid = (tok >= 0) & (page < ptw)
    req = token_to_req.long().unsqueeze(1).expand_as(tok)
    phys = block_table.long()[req, page.clamp(max=ptw - 1)]
    valid = valid & (phys >= 0) & (phys < num_dev + num_host)
    on_dev = phys < num_dev
    safe_dev = phys.clamp(0, num_dev - 1)
    safe_host = (phys - num_dev).clamp(0, max(num_host - 1, 0))

    out = torch.zeros_like(q)
    for h in range(kvh):
        k_d = k_cache[safe_dev, off, h].float()
        v_d = v_cache[safe_dev, off, h].float()
        if num_host:
            k_host, v_host = host_kv
            k_h = k_host[safe_host.cpu(), off.cpu(), h].to(DEV).float()
            v_h = v_host[safe_host.cpu(), off.cpu(), h].to(DEV).float()
            k_sel = torch.where(on_dev.unsqueeze(-1), k_d, k_h)
            v_sel = torch.where(on_dev.unsqueeze(-1), v_d, v_h)
        else:
            k_sel, v_sel = k_d, v_d
        heads = slice(h * group, (h + 1) * group)
        qh = q[:, heads, :].float()
        scores = torch.einsum("rgd,rtd->rgt", qh, k_sel) / (d**0.5)
        scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
        w = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        out[:, heads, :] = torch.einsum("rgt,rtd->rgd", w, v_sel).to(q.dtype)
    return out


def make_scene(rows, topk, num_dev, num_host, page_size=64, kvh=2, hq=24, d=256, seed=0):
    """Random device+host KV slabs, a shuffled page table, and a query batch selecting
    ``topk`` logical tokens per row out of the ``(num_dev + num_host) * page_size`` slots."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    total_pages = num_dev + num_host
    k_dev = torch.randn(num_dev, page_size, kvh, d, device=DEV, dtype=DTYPE, generator=g)
    v_dev = torch.randn(num_dev, page_size, kvh, d, device=DEV, dtype=DTYPE, generator=g)
    if num_host:
        k_host = host_empty((num_host, page_size, kvh, d))
        v_host = host_empty((num_host, page_size, kvh, d))
        k_host.copy_(torch.randn(num_host, page_size, kvh, d, device=DEV, generator=g).cpu())
        v_host.copy_(torch.randn(num_host, page_size, kvh, d, device=DEV, generator=g).cpu())
        host_kv = (k_host, v_host)
    else:
        host_kv = None
    block_table = torch.randperm(total_pages, device=DEV, generator=g)[None, :].to(torch.int32)
    q = torch.randn(rows, hq, d, device=DEV, dtype=DTYPE, generator=g)
    logical_indices = torch.randint(
        0, total_pages * page_size, (rows, topk), device=DEV, generator=g, dtype=torch.int64
    ).to(torch.int32)
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=DEV)
    return q, k_dev, v_dev, host_kv, logical_indices, block_table, token_to_req


# ---------------------------------------------------------------------------
# tiered_store_kv / gather_pages / zero_tier
# ---------------------------------------------------------------------------


def test_tiered_store_kv_all_vram_is_bit_exact():
    ps, h, d, n = 64, 2, 256, 8
    t = n * ps
    k = torch.randn(t, h * d, device=DEV, dtype=DTYPE)
    v = torch.randn_like(k)
    perm = torch.randperm(t, device=DEV).to(torch.int32)
    k_dev = torch.zeros(n, ps, h, d, device=DEV, dtype=DTYPE)
    v_dev = torch.zeros_like(k_dev)
    k_host = torch.zeros(0, ps, h, d, device=DEV, dtype=DTYPE)
    v_host = torch.zeros_like(k_host)
    tiered_store_kv(k, v, perm, (k_dev, v_dev), (k_host, v_host))
    torch.cuda.synchronize()

    ref = torch.zeros(t, h * d, device=DEV, dtype=DTYPE)
    ref[perm.long()] = k
    assert torch.equal(k_dev.view(t, h * d), ref)


def test_tiered_store_kv_device_and_host_is_bit_exact():
    ps, h, d, n, hh = 64, 2, 256, 4, 4
    t = (n + hh) * ps
    k = torch.randn(t, h * d, device=DEV, dtype=DTYPE)
    v = torch.randn_like(k)
    perm = torch.randperm(t, device=DEV).to(torch.int32)
    k_dev = torch.zeros(n, ps, h, d, device=DEV, dtype=DTYPE)
    v_dev = torch.zeros_like(k_dev)
    k_host = host_empty((hh, ps, h, d))
    v_host = host_empty((hh, ps, h, d))
    k_host.zero_()
    v_host.zero_()
    tiered_store_kv(k, v, perm, (k_dev, v_dev), (k_host, v_host))
    torch.cuda.synchronize()

    ref_k = torch.zeros(t, h * d, device=DEV, dtype=DTYPE)
    ref_k[perm.long()] = k
    got_k = torch.cat([k_dev.view(n * ps, h * d), k_host.view(hh * ps, h * d).to(DEV)])
    assert torch.equal(got_k, ref_k)


def test_tiered_store_kv_out_of_range_slots_are_dropped():
    ps, h, d, n = 8, 1, 16, 2
    t = n * ps
    k = torch.randn(t, h * d, device=DEV, dtype=DTYPE)
    v = torch.randn_like(k)
    perm = torch.arange(t, device=DEV, dtype=torch.int32)
    perm[0] = -1
    perm[1] = t + 5  # beyond both tiers
    k_dev = torch.zeros(n, ps, h, d, device=DEV, dtype=DTYPE)
    v_dev = torch.zeros_like(k_dev)
    k_host = torch.zeros(0, ps, h, d, device=DEV, dtype=DTYPE)
    tiered_store_kv(k, v, perm, (k_dev, v_dev), (k_host, k_host))
    torch.cuda.synchronize()
    flat = k_dev.view(t, h * d)
    assert torch.equal(flat[2:], k[2:])
    assert torch.equal(flat[:2], torch.zeros(2, h * d, device=DEV, dtype=DTYPE))


def test_gather_pages_matches_indexing_device_and_host():
    n, ps, h, d = 6, 32, 2, 64
    src_dev = torch.randn(n, ps, h, d, device=DEV, dtype=DTYPE)
    dst = torch.zeros(3, ps, h, d, device=DEV, dtype=DTYPE)
    pages = torch.tensor([5, 0, 2], device=DEV, dtype=torch.int32)
    gather_pages(src_dev, pages, dst)
    assert torch.equal(dst, src_dev[pages.long()])

    src_host = host_empty((n, ps, h, d))
    src_host.copy_(torch.randn(n, ps, h, d).to(DTYPE))
    dst2 = torch.zeros(3, ps, h, d, device=DEV, dtype=DTYPE)
    gather_pages(src_host, pages, dst2)
    assert torch.equal(dst2, src_host[pages.cpu().long()].to(DEV))


def test_zero_tier_stream_ordered_after_store():
    n, ps, h, d = 2, 16, 1, 8
    t = n * ps
    k = torch.randn(t, h * d, device=DEV, dtype=DTYPE)
    perm = torch.arange(t, device=DEV, dtype=torch.int32)
    k_dev = torch.zeros(n, ps, h, d, device=DEV, dtype=DTYPE)
    v_dev = torch.zeros_like(k_dev)
    k_host = torch.zeros(0, ps, h, d, device=DEV, dtype=DTYPE)
    tiered_store_kv(k, k, perm, (k_dev, v_dev), (k_host, k_host))
    zero_tier(k_dev)
    torch.cuda.synchronize()
    assert torch.equal(k_dev, torch.zeros_like(k_dev))


# ---------------------------------------------------------------------------
# qsa_sparse_paged_attention over the host tier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 2, 5, 8])
def test_host_tier_attention_matches_eager_decode_like(rows):
    q, kd, vd, host_kv, idx, bt, t2r = make_scene(rows, topk=64, num_dev=8, num_host=8, seed=rows)
    ref = reference_attention(q, kd, vd, host_kv, idx, bt, t2r)
    got = qsa_sparse_paged_attention(q, kd, vd, idx, bt, t2r, host_kv=host_kv)
    torch.testing.assert_close(got.float(), ref.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("rows", [64, 512])
def test_host_tier_attention_matches_eager_prefill_like(rows):
    # head_dim=256 on the wide prefill tiles: guards the tiered path's shared-memory footprint.
    q, kd, vd, host_kv, idx, bt, t2r = make_scene(
        rows, topk=256, num_dev=8, num_host=8, d=256, seed=rows + 1
    )
    ref = reference_attention(q, kd, vd, host_kv, idx, bt, t2r)
    got = qsa_sparse_paged_attention(q, kd, vd, idx, bt, t2r, host_kv=host_kv)
    torch.testing.assert_close(got.float(), ref.float(), atol=3e-2, rtol=3e-2)


def test_all_vram_out_of_range_pages_are_masked_zero_copy():
    q, kd, vd, _, idx, bt, t2r = make_scene(4, topk=32, num_dev=8, num_host=0, seed=7)
    # send some selections through pages beyond the table width -> masked, must not crash
    idx[:, -1] = kd.shape[0] * kd.shape[1] + 10_000
    got = qsa_sparse_paged_attention(q, kd, vd, idx, bt, t2r, host_kv=None)
    assert torch.isfinite(got.float()).all()
    ref = reference_attention(q, kd, vd, None, idx, bt, t2r)
    torch.testing.assert_close(got.float(), ref.float(), atol=3e-2, rtol=3e-2)


def test_cuda_graph_replay_matches_eager_no_reallocation():
    q, kd, vd, host_kv, idx, bt, t2r = make_scene(4, topk=64, num_dev=8, num_host=8, seed=99)
    ref = reference_attention(q, kd, vd, host_kv, idx, bt, t2r)
    out = torch.empty_like(q)

    # warm-up run required before capture (Triton autotune/compile must not happen in-graph)
    qsa_sparse_paged_attention(q, kd, vd, idx, bt, t2r, out=out, host_kv=host_kv)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qsa_sparse_paged_attention(q, kd, vd, idx, bt, t2r, out=out, host_kv=host_kv)

    k_ptr_before, v_ptr_before = kd.data_ptr(), vd.data_ptr()
    graph.replay()
    torch.cuda.synchronize()
    assert kd.data_ptr() == k_ptr_before and vd.data_ptr() == v_ptr_before
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


def test_eviction_reload_ring_reuse_reflects_latest_write():
    """Writing new tokens into a physical page previously freed/reused (ring/slab semantics)
    must be visible to attention immediately -- no stale device or host copy lingers."""
    ps, h, d, n, hh = 16, 1, 32, 2, 2
    q = torch.randn(1, h, d, device=DEV, dtype=DTYPE)
    k_dev = torch.zeros(n, ps, h, d, device=DEV, dtype=DTYPE)
    v_dev = torch.zeros_like(k_dev)
    k_host = host_empty((hh, ps, h, d))
    v_host = host_empty((hh, ps, h, d))
    k_host.zero_()
    v_host.zero_()

    # first tenant of host page 0 (physical page n)
    t = ps
    k1 = torch.randn(t, h * d, device=DEV, dtype=DTYPE)
    perm = torch.arange(t, device=DEV, dtype=torch.int32) + n * ps
    tiered_store_kv(k1, k1, perm, (k_dev, v_dev), (k_host, v_host))
    torch.cuda.synchronize()

    # ring reuse: a new tenant overwrites the same physical host page
    k2 = torch.randn(t, h * d, device=DEV, dtype=DTYPE)
    tiered_store_kv(k2, k2, perm, (k_dev, v_dev), (k_host, v_host))
    torch.cuda.synchronize()
    assert torch.equal(k_host.view(hh * ps, h * d)[:ps].to(DEV), k2)
    assert not torch.equal(k_host.view(hh * ps, h * d)[:ps].to(DEV), k1)

    bt = torch.tensor([[n]], device=DEV, dtype=torch.int32)  # points at the reused host page
    idx = torch.arange(ps, device=DEV, dtype=torch.int32)[None, :]
    t2r = torch.zeros(1, dtype=torch.int32, device=DEV)
    got = qsa_sparse_paged_attention(q, k_dev, v_dev, idx, bt, t2r, host_kv=(k_host, v_host))
    ref = reference_attention(q, k_dev, v_dev, (k_host, v_host), idx, bt, t2r)
    torch.testing.assert_close(got.float(), ref.float(), atol=3e-2, rtol=3e-2)


# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------


def test_fault_stale_physical_page_in_table_is_detected():
    q, kd, vd, host_kv, idx, bt, t2r = make_scene(4, topk=64, num_dev=8, num_host=8, seed=11)
    ref = reference_attention(q, kd, vd, host_kv, idx, bt, t2r)
    bad_bt = bt.clone()
    bad_bt[0, 0] = (bad_bt[0, 0] + 1) % bt.max().add(1).item()
    got = qsa_sparse_paged_attention(q, kd, vd, idx, bad_bt, t2r, host_kv=host_kv)
    assert not torch.allclose(got.float(), ref.float(), atol=3e-2, rtol=3e-2)


def test_fault_wrong_host_strides_raises():
    q, kd, vd, host_kv, idx, bt, t2r = make_scene(2, topk=16, num_dev=4, num_host=4, seed=13)
    k_host, v_host = host_kv
    # same shape as k_host, but padded/narrowed so its strides no longer match k_cache's.
    padded = host_empty((*k_host.shape[:-1], k_host.shape[-1] * 2))
    padded.zero_()
    padded[..., : k_host.shape[-1]].copy_(k_host)
    wrong_strides = padded[..., : k_host.shape[-1]]
    assert wrong_strides.shape == k_host.shape and wrong_strides.stride() != k_host.stride()
    with pytest.raises(ValueError, match="strides"):
        qsa_sparse_paged_attention(q, kd, vd, idx, bt, t2r, host_kv=(wrong_strides, v_host))


def test_fault_mismatched_host_page_layout_raises():
    q, kd, vd, host_kv, idx, bt, t2r = make_scene(2, topk=16, num_dev=4, num_host=4, seed=17)
    k_host, v_host = host_kv
    wrong_shape = host_empty((k_host.shape[0], k_host.shape[1] * 2, *k_host.shape[2:]))
    with pytest.raises(ValueError, match="layout"):
        qsa_sparse_paged_attention(q, kd, vd, idx, bt, t2r, host_kv=(wrong_shape, v_host))


def test_fault_incomplete_prefetch_mismatch_is_detected():
    n, hh, ps, h, d = 4, 4, 32, 2, 64
    q, kd, vd, host_kv, idx, bt, t2r = make_scene(
        3, topk=64, num_dev=n, num_host=hh, page_size=ps, kvh=h, d=d, seed=23
    )
    ref = reference_attention(q, kd, vd, host_kv, idx, bt, t2r)

    # stage only half the host pages that the table actually references into device
    # scratch, then attend against host_kv anyway with a table that assumes full staging.
    k_host, v_host = host_kv
    host_pages_used = torch.unique(bt[bt >= n]) - n
    staged = host_pages_used[: max(1, len(host_pages_used) // 2)]
    stale_k_host = host_empty(k_host.shape)
    stale_v_host = host_empty(v_host.shape)
    stale_k_host.zero_()
    stale_v_host.zero_()
    gather_pages(k_host, staged.to(torch.int32), stale_k_host[: len(staged)])
    gather_pages(v_host, staged.to(torch.int32), stale_v_host[: len(staged)])
    # stale_{k,v}_host now has real data only for the first len(staged) rows; the rest of the
    # rows the table still points at are wrong/zero -> output must diverge from the reference.

    remap = {int(p): i for i, p in enumerate(staged.tolist())}
    remapped_bt = bt.clone()
    host_mask = remapped_bt >= n
    remapped_bt[host_mask] = torch.tensor(
        [n + remap.get(int(p) - n, 0) for p in remapped_bt[host_mask].tolist()],
        device=DEV,
        dtype=torch.int32,
    )
    got = qsa_sparse_paged_attention(
        q, kd, vd, idx, remapped_bt, t2r, host_kv=(stale_k_host, stale_v_host)
    )
    assert not torch.allclose(got.float(), ref.float(), atol=3e-2, rtol=3e-2)


def test_fault_page_id_beyond_both_tiers_is_masked_not_crashed():
    q, kd, vd, host_kv, idx, bt, t2r = make_scene(2, topk=16, num_dev=4, num_host=4, seed=29)
    ref = reference_attention(q, kd, vd, host_kv, idx, bt, t2r)
    bad_bt = bt.clone()
    bad_bt[0, 0] = bt.numel() + 999  # past both tiers entirely
    got = qsa_sparse_paged_attention(q, kd, vd, idx, bad_bt, t2r, host_kv=host_kv)
    assert torch.isfinite(got.float()).all()
    assert not torch.allclose(got.float(), ref.float(), atol=3e-2, rtol=3e-2)


def test_fault_wrong_device_pages_boundary_is_detected():
    q, kd, vd, host_kv, idx, bt, t2r = make_scene(3, topk=32, num_dev=8, num_host=8, seed=31)
    ref = reference_attention(q, kd, vd, host_kv, idx, bt, t2r)
    # caller passes a k_cache/v_cache that only covers half the real device slab: the
    # device/host boundary the kernel infers from k_cache.shape[0] no longer matches the
    # table, which still names physical pages against the true (larger) device pool.
    shrunk_k, shrunk_v = kd[: kd.shape[0] // 2].contiguous(), vd[: vd.shape[0] // 2].contiguous()
    got = qsa_sparse_paged_attention(q, shrunk_k, shrunk_v, idx, bt, t2r, host_kv=host_kv)
    assert not torch.allclose(got.float(), ref.float(), atol=3e-2, rtol=3e-2)


# ---------------------------------------------------------------------------
# MHAKVCache / QSAKVCache host_pages wiring
# ---------------------------------------------------------------------------


def test_mha_kv_cache_host_pages_api():
    pool = MHAKVCache(
        num_kv_heads=2,
        num_layers=2,
        head_dim=32,
        num_pages=8,
        page_size=16,
        dtype=DTYPE,
        device=DEV,
        host_pages=4,
    )
    assert pool.num_device_pages == 4
    k, v = pool.host_kv(0)
    assert k.shape == (4, 16, 2, 32) and v.shape == (4, 16, 2, 32)

    tokens = 16  # fills exactly page 0, addressed inside the device tier
    kk = torch.randn(tokens, 2 * 32, device=DEV, dtype=DTYPE)
    vv = torch.randn_like(kk)
    out_loc = torch.arange(tokens, device=DEV, dtype=torch.int32)
    pool.store_kv(kk, vv, out_loc, layer_id=0)
    torch.cuda.synchronize()
    assert torch.equal(pool.k_cache(0)[0].view(tokens, -1), kk)

    host_tokens = 16
    kk2 = torch.randn(host_tokens, 2 * 32, device=DEV, dtype=DTYPE)
    out_loc2 = torch.arange(host_tokens, device=DEV, dtype=torch.int32) + pool.num_device_pages * 16
    pool.store_kv(kk2, kk2, out_loc2, layer_id=0)
    torch.cuda.synchronize()
    assert torch.equal(pool.host_kv(0)[0][0].view(host_tokens, -1).to(DEV), kk2)


def test_mha_kv_cache_all_vram_has_no_host_tier():
    pool = MHAKVCache(
        num_kv_heads=2,
        num_layers=1,
        head_dim=32,
        num_pages=4,
        page_size=16,
        dtype=DTYPE,
        device=DEV,
        host_pages=0,
    )
    assert pool.host_kv(0) is None
    assert pool.num_device_pages == 4


def test_qsa_kv_cache_host_pages_api():
    pool = QSAKVCache(
        num_kv_heads=2,
        num_layers=2,
        head_dim=32,
        num_pages=8,
        page_size=16,
        dtype=DTYPE,
        device=DEV,
        index_head_dim=16,
        num_index_layers=2,
        index_ratio=4,
        num_req_slots=2,
        layer_ids=(0, 1),
        host_pages=4,
    )
    assert pool.num_device_pages == 4
    k, v = pool.host_kv(0)
    assert k.shape == (4, 16, 2, 32) and v.shape == (4, 16, 2, 32)
    assert pool.host_staging[0].shape == (4, 16, 2, 32)
    pool.clear_mtp_slot()  # must not raise even with no mtp layer configured


def test_qsa_kv_cache_rejects_host_pages_with_compressed_format():
    with pytest.raises(NotImplementedError):
        QSAKVCache(
            num_kv_heads=2,
            num_layers=2,
            head_dim=32,
            num_pages=8,
            page_size=16,
            dtype=DTYPE,
            device=DEV,
            index_head_dim=16,
            num_index_layers=2,
            index_ratio=4,
            num_req_slots=2,
            layer_ids=(0, 1),
            host_pages=4,
            kv_format="turbo4",
        )


def _logical_contents(pool):
    """Every logical page's K/V/index rows, read through the rebalancer's page map."""
    total = pool.page_map.shape[0]
    kv = torch.cat([pool._pool._kv_buffer, pool._pool._kv_host.to(DEV)], dim=2)
    per_page = pool._page_size // pool._index_ratio
    cmp = pool._cmp_k_buffer[:, : total * per_page].unflatten(1, (total, per_page))
    phys = pool.page_map.long()
    return kv[:, :, phys].clone(), cmp[:, phys].clone()


def test_rebalance_promotes_hot_ram_pages_and_preserves_logical_contents():
    torch.manual_seed(7)
    pool = QSAKVCache(
        num_kv_heads=2,
        num_layers=2,
        head_dim=32,
        num_pages=8,
        page_size=16,
        dtype=DTYPE,
        device=DEV,
        index_head_dim=16,
        num_index_layers=2,
        index_ratio=4,
        num_req_slots=2,
        layer_ids=(0, 1),
        host_pages=4,
    )
    pool._pool._kv_buffer.normal_()
    pool._pool._kv_host.copy_(torch.randn_like(pool._pool._kv_host, dtype=torch.float32))
    pool._cmp_k_buffer.normal_()
    before = _logical_contents(pool)
    pool.page_heat[:8] = torch.tensor([9, 0, 1, 0, 40, 0, 30, 2], device=DEV, dtype=torch.int32)
    pool.rebalance(4)
    torch.cuda.synchronize()
    after = _logical_contents(pool)
    assert torch.equal(before[0], after[0]) and torch.equal(before[1], after[1])
    # RAM pages 4 and 6 were hot: they now sit on the device; the cold device pages moved out.
    assert int(pool.page_map[4]) < 4 and int(pool.page_map[6]) < 4
    assert int(pool.page_map[0]) == 0  # hot device page stays
    assert int(pool.page_map[5]) >= 4  # cold RAM page stays
    assert sorted(pool.page_map.tolist()) == list(range(8))
    # Second pass with no heat is a no-op for contents.
    pool.rebalance(4)
    assert torch.equal(_logical_contents(pool)[0], before[0])


def test_fp8_ram_tier_store_attend_stage_and_swap_track_bf16_reference():
    torch.manual_seed(11)
    ps, h, d, hq, n_dev, n_host = 64, 2, 256, 24, 8, 8
    tokens = (n_dev + n_host) * ps
    k = torch.randn(tokens, h * d, device=DEV, dtype=DTYPE)
    v = torch.randn_like(k)
    page_table = torch.randperm(n_dev + n_host, device=DEV).to(torch.int32)
    slots = (page_table.long()[:, None] * ps + torch.arange(ps, device=DEV)).reshape(-1)
    ref_k = torch.zeros(n_dev + n_host, ps, h, d, device=DEV, dtype=DTYPE)
    ref_v = torch.zeros_like(ref_k)
    ref_k.view(tokens, -1)[slots] = k
    ref_v.view(tokens, -1)[slots] = v
    kd = torch.zeros(n_dev, ps, h, d, device=DEV, dtype=DTYPE)
    vd = torch.zeros_like(kd)
    fp8 = torch.float8_e4m3fn
    kh = registered_host_empty((n_host, ps, h, d), fp8)
    vh = registered_host_empty((n_host, ps, h, d), fp8)
    tiered_store_kv(k, v, slots.to(torch.int32), (kd, vd), (kh, vh))
    torch.cuda.synchronize()
    assert torch.equal(kh.to(DTYPE), ref_k[n_dev:].cpu().to(fp8).to(DTYPE))
    rows = 4
    q = torch.randn(rows, hq, d, device=DEV, dtype=DTYPE)
    idx = torch.stack([torch.randperm(tokens, device=DEV)[:2048] for _ in range(rows)])
    idx = idx.to(torch.int32).contiguous()
    bt = page_table[None, :].contiguous()
    t2r = torch.zeros(rows, dtype=torch.int32, device=DEV)
    ref = qsa_sparse_paged_attention(q, ref_k, ref_v, idx, bt, t2r)
    got = qsa_sparse_paged_attention(q, kd, vd, idx, bt, t2r, host_kv=(kh, vh))
    assert torch.allclose(got.float(), ref.float(), atol=5e-2, rtol=5e-2)
    # Staged (prefill) path widens FP8 into the BF16 staging slab: the same values reach the
    # dot, but through a differently compiled tile, so compare numerically.
    pages = torch.arange(n_host, device=DEV)
    sk = torch.empty(n_host, ps, h, d, device=DEV, dtype=DTYPE)
    sv = torch.empty_like(sk)
    gather_pages(kh, pages, sk)
    gather_pages(vh, pages, sv)
    staged = qsa_sparse_paged_attention(q, kd, vd, idx, bt, t2r, host_kv=(sk, sv))
    assert torch.allclose(staged.float(), got.float(), atol=1e-2, rtol=1e-2)
    # A BF16 <-> FP8 swap moves each page to the other tier with the right cast.
    a = kd.clone().unsqueeze(0)
    b = registered_host_empty((1, n_host, ps, h, d), fp8)
    b.copy_(kh.unsqueeze(0))
    swap_pages(
        a,
        b,
        torch.tensor([0], device=DEV),
        torch.tensor([1], device=DEV),
        torch.ones(1, dtype=torch.int32, device=DEV),
    )
    torch.cuda.synchronize()
    assert torch.equal(a[0, 0], kh[1].to(DEV).to(DTYPE))
    assert torch.equal(b[0, 1], kd[0].cpu().to(fp8))


@pytest.mark.parametrize("book", ["turbo4", "turbo3"])
def test_turbo_ram_tier_store_decode_and_attention_track_reference(book):
    from freetoken.kernel.triton import turbo_kv as tk
    from freetoken.kernel.triton.qsa.tiered import turbo_inverse_rotation, turbo_pages_to_bf16

    torch.manual_seed(5)
    ps, h, d, hq = 64, 2, 256, 24
    pool = MHAKVCache(
        num_kv_heads=h,
        num_layers=1,
        head_dim=d,
        num_pages=8,
        page_size=ps,
        dtype=DTYPE,
        device=DEV,
        host_pages=4,
        host_dtype=book,
    )
    tokens = 8 * ps
    k = torch.randn(tokens, h * d, device=DEV, dtype=DTYPE)
    v = torch.randn_like(k)
    loc = torch.arange(tokens, device=DEV, dtype=torch.int32)
    pool.store_kv(k, v, loc, 0)
    torch.cuda.synchronize()
    kc, kn, vc, vn = pool.host_turbo(0)
    ref_codes, ref_norm = tk.quantize(k[4 * ps :].reshape(-1, d), book)
    assert torch.equal(kc.reshape(-1, kc.shape[-1]).to(DEV), ref_codes)
    assert torch.equal(pool.k_cache(0).reshape(4 * ps, -1), k[: 4 * ps])
    stage_k = torch.zeros(4, ps, h, d, device=DEV, dtype=DTYPE)
    stage_v = torch.zeros_like(stage_k)
    cent = tk._book(torch.device(DEV), book)[0].float()
    rot = turbo_inverse_rotation(torch.device(DEV))
    ids = torch.arange(4, device=DEV, dtype=torch.int32)
    turbo_pages_to_bf16(kc, kn, cent, rot, ids, ids, stage_k, book, ps)
    turbo_pages_to_bf16(vc, vn, cent, rot, ids, ids, stage_v, book, ps)
    oracle = tk.decode(ref_codes, ref_norm, book).reshape(4, ps, h, d)
    rel = (stage_k.float() - oracle.float()).norm() / oracle.float().norm()
    assert rel < 1e-2
    q = torch.randn(3, hq, d, device=DEV, dtype=DTYPE)
    idx = torch.stack([torch.randperm(tokens, device=DEV)[:1024] for _ in range(3)])
    idx = idx.to(torch.int32).contiguous()
    bt = torch.arange(8, device=DEV, dtype=torch.int32)[None, :].contiguous()
    t2r = torch.zeros(3, dtype=torch.int32, device=DEV)
    full_k = torch.cat([pool.k_cache(0), oracle.to(DTYPE)])
    full_v = torch.cat(
        [
            pool.v_cache(0),
            tk.decode(*tk.quantize(v[4 * ps :].reshape(-1, d), book), book)
            .reshape(4, ps, h, d)
            .to(DTYPE),
        ]
    )
    ref = qsa_sparse_paged_attention(q, full_k, full_v, idx, bt, t2r)
    got = qsa_sparse_paged_attention(
        q, pool.k_cache(0), pool.v_cache(0), idx, bt, t2r, host_kv=(stage_k, stage_v)
    )
    assert torch.allclose(got.float(), ref.float(), atol=3e-2, rtol=3e-2)
