"""Wiring pin for the fused decode: attending on turbo codes must reproduce attending on the
*decoded* values, because the codec's own error is already pinned in
tests/kvcache/test_turbo_kv.py. Comparing against the decoded cache isolates the one thing this
test can break -- the rotated-domain bookkeeping (pre-rotated Q, no per-tile inverse, output
rotated back once) -- from quantization error.
"""


import pytest
import torch

from freetoken.kernel.triton import turbo_kv as tk

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _splits(batch, num_q_heads, max_kv_splits, head_dim, device):
    return (
        torch.empty(batch, num_q_heads, max_kv_splits, head_dim, dtype=torch.float32, device=device),
        torch.empty(batch, num_q_heads, max_kv_splits, dtype=torch.float32, device=device),
        torch.full((batch,), max_kv_splits, dtype=torch.int32, device=device),
    )


@pytest.mark.parametrize("book", list(tk.BOOKS))
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
    indptr = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0)), dtype=torch.int32, device=device)
    indices = torch.arange(total, dtype=torch.int32, device=device)
    q_positions = torch.tensor([s - 1 for s in seq_lens], dtype=torch.int64, device=device)

    kc, kn = tk.quantize(k.reshape(-1, head_dim), book)
    vc, vn = tk.quantize(v.reshape(-1, head_dim), book)
    k_hat = tk.decode(kc, kn, book).reshape(total, kv_heads, head_dim).to(torch.bfloat16)
    v_hat = tk.decode(vc, vn, book).reshape(total, kv_heads, head_dim).to(torch.bfloat16)

    logits, lse, splits = _splits(batch, q_heads, max_kv_splits, head_dim, device)
    want = decode_paged_attention(
        q, k_hat, v_hat, indptr, indices, q_positions, logits, lse, splits,
        max_kv_splits, sm_scale,
    )

    # same values, reached the fused way: codes + norm, Q pre-rotated, output rotated back once
    q_rot = tk.rotate(q.reshape(-1, head_dim)).reshape(q.shape).to(torch.bfloat16)
    codes_k = kc.reshape(total, kv_heads, -1).contiguous()
    codes_v = vc.reshape(total, kv_heads, -1).contiguous()
    norm_k = kn.reshape(total, kv_heads, 1).contiguous()
    norm_v = vn.reshape(total, kv_heads, 1).contiguous()
    cent = torch.tensor(
        tk.CENTROIDS_3 if book == "turbo3" else tk.CENTROIDS_4, device=device, dtype=torch.float32
    )
    logits2, lse2, splits2 = _splits(batch, q_heads, max_kv_splits, head_dim, device)
    got_rot = decode_paged_attention(
        q_rot, codes_k, codes_v, indptr, indices, q_positions, logits2, lse2, splits2,
        max_kv_splits, sm_scale,
        turbo={"k_norm": norm_k, "v_norm": norm_v, "cent": cent, "book3": book == "turbo3"},
    )
    got = tk.inv_rotate(got_rot.reshape(-1, head_dim)).reshape(got_rot.shape).to(torch.bfloat16)

    assert torch.isfinite(got.float()).all()
    diff = (got.float() - want.float()).abs()
    rel = diff.max().item()
    assert rel < 2e-2, f"{book} @{q_heads}/{kv_heads}: fused decode drifted by {rel:.4f}"
    cos = torch.nn.functional.cosine_similarity(
        got.float().flatten(), want.float().flatten(), dim=0
    ).item()
    assert cos > 0.9999, f"{book}: fused decode is a different attention ({cos})"
