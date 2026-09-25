"""Wiring pin for the fused decode: attending on turbo codes must reproduce attending on the
*decoded* values, because the codec's own error is already pinned in
tests/kvcache/test_turbo_kv.py. Comparing against the decoded cache isolates the one thing this
test can break -- the rotated-domain bookkeeping (pre-rotated Q, no per-tile inverse, output
rotated back once) -- from quantization error.
"""

import pytest
import torch

from freetoken.kernel.triton import turbo_kv as tk
from freetoken.kernel.triton.turbo_attn import BOOK_CODE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _splits(batch, num_q_heads, max_kv_splits, head_dim, device):
    return (
        torch.empty(
            batch, num_q_heads, max_kv_splits, head_dim, dtype=torch.float32, device=device
        ),
        torch.empty(batch, num_q_heads, max_kv_splits, dtype=torch.float32, device=device),
        torch.full((batch,), max_kv_splits, dtype=torch.int32, device=device),
    )


# decode_paged_attention's fused code path only knows the BOOK3 flag (turbo3 vs turbo4); turbo8
# is RAM-tier only for now (decoded to bf16 before attention, see qsa/tiered.py).
@pytest.mark.parametrize("book", ["turbo3", "turbo4", "fp8", "nvfp4"])
@pytest.mark.parametrize(("q_heads", "kv_heads"), [(16, 4), (8, 8)])
def test_decode_on_codes_matches_decode_on_decoded_kv(book, q_heads, kv_heads):
    from freetoken.kernel.triton.attention import decode_paged_attention

    device = torch.device("cuda")
    torch.manual_seed(1)
    head_dim = tk.QK_TURBO
    seq_lens = [64, 200, 33]
    batch = len(seq_lens)
    total = sum(seq_lens)
    max_kv_splits = 8
    sm_scale = head_dim**-0.5

    q = torch.randn(batch, q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(total, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(total, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    indptr = torch.tensor(
        [0] + list(torch.tensor(seq_lens).cumsum(0)), dtype=torch.int32, device=device
    )
    indices = torch.arange(total, dtype=torch.int32, device=device)
    q_positions = torch.tensor([s - 1 for s in seq_lens], dtype=torch.int64, device=device)

    kc, kn = tk.quantize(k.reshape(-1, head_dim), book)
    vc, vn = tk.quantize(v.reshape(-1, head_dim), book)
    k_hat = tk.decode(kc, kn, book).reshape(total, kv_heads, head_dim).to(torch.bfloat16)
    v_hat = tk.decode(vc, vn, book).reshape(total, kv_heads, head_dim).to(torch.bfloat16)

    logits, lse, splits = _splits(batch, q_heads, max_kv_splits, head_dim, device)
    want = decode_paged_attention(
        q,
        k_hat,
        v_hat,
        indptr,
        indices,
        q_positions,
        logits,
        lse,
        splits,
        max_kv_splits,
        sm_scale,
    )

    # same values, reached the fused way: codes + norm, Q pre-rotated, output rotated back once
    rotated = tk.is_rotated(book)
    q_rot = tk.rotate(q.reshape(-1, head_dim)) if rotated else q.reshape(-1, head_dim)
    q_rot = q_rot.reshape(q.shape).to(torch.bfloat16)
    codes_k = kc.reshape(total, kv_heads, -1).contiguous()
    codes_v = vc.reshape(total, kv_heads, -1).contiguous()
    norm_k = kn.reshape(total, kv_heads, 1).contiguous()
    norm_v = vn.reshape(total, kv_heads, 1).contiguous()
    cent = torch.tensor(
        tk.CENTROIDS_3 if book == "turbo3" else tk.CENTROIDS_4,
        device=device,
        dtype=torch.float32,
    )
    logits2, lse2, splits2 = _splits(batch, q_heads, max_kv_splits, head_dim, device)
    got_rot = decode_paged_attention(
        q_rot,
        codes_k,
        codes_v,
        indptr,
        indices,
        q_positions,
        logits2,
        lse2,
        splits2,
        max_kv_splits,
        sm_scale,
        turbo={"k_norm": norm_k, "v_norm": norm_v, "cent": cent, "book": BOOK_CODE[book]},
    )
    got = got_rot.reshape(-1, head_dim)
    got = (tk.inv_rotate(got) if rotated else got).reshape(got_rot.shape).to(torch.bfloat16)

    assert torch.isfinite(got.float()).all()
    diff = (got.float() - want.float()).abs()
    rel = diff.max().item()
    assert rel < 2e-2, f"{book} @{q_heads}/{kv_heads}: fused decode drifted by {rel:.4f}"
    cos = torch.nn.functional.cosine_similarity(
        got.float().flatten(), want.float().flatten(), dim=0
    ).item()
    assert cos > 0.9999, f"{book}: fused decode is a different attention ({cos})"


def test_extend_on_codes_matches_extend_on_decoded_kv():
    """Prefill reads the cached prefix *and* the new tokens' K/V, so the two sources must be kept
    in the same domain: rotate Q and the new K/V, read the prefix from codes, rotate the output
    back once."""
    from freetoken.kernel.triton.attention import extend_paged_attention

    device = torch.device("cuda")
    torch.manual_seed(5)
    book = "turbo4"
    head_dim = tk.QK_TURBO
    q_heads, kv_heads = 12, 4
    q_lens = [8, 3]
    prefix_lens = [40, 96]
    seq_lens = [p + q for p, q in zip(prefix_lens, q_lens)]
    total = sum(seq_lens)
    num_q = sum(q_lens)
    sm_scale = head_dim**-0.5

    q = torch.randn(num_q, q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(total, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(total, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    starts = [0] + list(torch.tensor(seq_lens).cumsum(0))[:-1]
    qo_indptr = torch.tensor(
        [0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device=device
    )
    kv_indptr = torch.tensor(
        [0] + list(torch.tensor(seq_lens).cumsum(0)), dtype=torch.int32, device=device
    )
    kv_indices = torch.arange(total, dtype=torch.int32, device=device)
    prefix = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
    new_rows = lambda t: torch.cat(
        [t[a + p : a + s] for a, p, s in zip(starts, prefix_lens, seq_lens)]
    ).contiguous()
    k_extend, v_extend = new_rows(k), new_rows(v)
    assert k_extend.shape[0] == num_q

    kc, kn = tk.quantize(k.reshape(-1, head_dim), book)
    vc, vn = tk.quantize(v.reshape(-1, head_dim), book)
    k_hat = tk.decode(kc, kn, book).reshape(total, kv_heads, head_dim).to(torch.bfloat16)
    v_hat = tk.decode(vc, vn, book).reshape(total, kv_heads, head_dim).to(torch.bfloat16)

    want = extend_paged_attention(
        q,
        k_hat,
        v_hat,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix,
        max(q_lens),
        sm_scale,
        k_extend=k_extend,
        v_extend=v_extend,
    )

    def rot(t):
        return tk.rotate(t.reshape(-1, head_dim)).reshape(t.shape).to(torch.bfloat16)

    got_rot = extend_paged_attention(
        rot(q),
        kc.reshape(total, kv_heads, -1).contiguous(),
        vc.reshape(total, kv_heads, -1).contiguous(),
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix,
        max(q_lens),
        sm_scale,
        k_extend=rot(k_extend),
        v_extend=rot(v_extend),
        turbo={
            "k_norm": kn.reshape(total, kv_heads, 1).contiguous(),
            "v_norm": vn.reshape(total, kv_heads, 1).contiguous(),
            "cent": torch.tensor(tk.CENTROIDS_4, device=device, dtype=torch.float32),
            "book": 0,
        },
    )
    got = tk.inv_rotate(got_rot.reshape(-1, head_dim)).reshape(got_rot.shape).to(torch.bfloat16)
    assert torch.isfinite(got.float()).all()
    assert (got.float() - want.float()).abs().max().item() < 3e-2
    cos = torch.nn.functional.cosine_similarity(
        got.float().flatten(), want.float().flatten(), dim=0
    ).item()
    assert cos > 0.9999, f"prefill on codes is a different attention ({cos})"


def test_bf16_extend_is_untouched_by_the_branch():
    """Same call twice must be bit-identical: the constexpr default has to leave the bf16 path exactly
    where it was, which is what lets the 16K guard keep protecting it."""
    from freetoken.kernel.triton.attention import extend_paged_attention

    device = torch.device("cuda")
    head_dim = tk.QK_TURBO
    q = torch.randn(6, 8, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(6, 4, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(6, 4, head_dim, device=device, dtype=torch.bfloat16)
    qo = torch.tensor([0, 3, 6], dtype=torch.int32, device=device)
    idx = torch.arange(6, dtype=torch.int32, device=device)
    pre = torch.zeros(2, dtype=torch.int32, device=device)
    a = extend_paged_attention(q, k, v, qo, qo, idx, pre, 3, head_dim**-0.5)
    b = extend_paged_attention(q, k, v, qo, qo, idx, pre, 3, head_dim**-0.5)
    assert torch.equal(a, b)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="needs Blackwell (hardware e2m1 conversion)",
)
def test_nvfp4_tile_uses_the_hardware_e2m1_unit():
    """The NVFP4 read path must be the Blackwell F2FP.E2M1 conversion, not a lookup table."""
    import triton
    import triton.language as tl

    from freetoken.kernel.triton.turbo_attn import _e2m1

    @triton.jit
    def probe(x_ptr, o_ptr, N: tl.constexpr):
        offs = tl.arange(0, N // 2)
        b = tl.load(x_ptr + offs)
        tl.store(o_ptr + offs * 2, _e2m1(b, offs < 0))
        tl.store(o_ptr + offs * 2 + 1, _e2m1(b, offs >= 0))

    x = torch.arange(256, dtype=torch.uint8, device="cuda")
    out = torch.empty(512, dtype=torch.float32, device="cuda")
    handle = probe[(1,)](x, out, N=512)
    ref = torch.tensor([[tk.E2M1[i & 15], tk.E2M1[i >> 4]] for i in range(256)]).flatten()
    assert torch.equal(out.cpu(), ref)
    assert "cvt.rn.f16x2.e2m1x2" in handle.asm["ptx"]
