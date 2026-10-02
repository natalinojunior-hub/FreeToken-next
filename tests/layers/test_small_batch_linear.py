import pytest
import torch
import torch.nn.functional as F

from freetoken.layers.quantization.linear.unquantized import small_batch_linear


def test_small_batch_linear_is_contiguous_and_matches():
    """CPU activations fall through to F.linear (the split-K kernel is CUDA/bf16 only);
    consumers (hc_silu) need row-major output."""
    w = torch.randn(48, 64, dtype=torch.bfloat16)
    for rows in (1, 2, 5, 8, 9):
        x = torch.randn(rows, 64, dtype=torch.bfloat16)
        y = small_batch_linear(x, w)
        assert y.is_contiguous()
        torch.testing.assert_close(y.float(), (x @ w.T).float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("n,k", [(512, 2560), (128, 2560), (48, 2560), (336, 10240)])
@pytest.mark.parametrize("m", [2, 3, 8])
def test_cuda_dispatch_matches_fp32(n, k, m):
    """The 2..8 band is dispatched per shape (split-K for N<256, cuBLAS F.linear above); both
    must stay a faithful bf16 GEMV and return row-major output. M=3 is the MTP verify size."""
    g = torch.Generator(device="cuda").manual_seed(3)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    y = small_batch_linear(x, w)
    ref = x.float() @ w.float().T
    assert y.shape == (m, n) and y.dtype == torch.bfloat16 and y.is_contiguous()
    rel = ((y.float() - ref).norm() / ref.norm()).item()
    assert rel < 5e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("n,k", [(1, 2048), (64, 2048), (256, 2048), (512, 2560), (2048, 4096)])
def test_m1_never_uses_small_band(n, k, monkeypatch):
    """Decode and verify share a per-weight policy and are bitwise row-invariant."""
    import freetoken.layers.quantization.linear.unquantized as uq

    monkeypatch.setenv("FREETOKEN_ROW_INVARIANT_LINEAR", "1")
    g = torch.Generator(device="cuda").manual_seed(4)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    x = torch.randn(8, k, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=g)
    single = torch.cat([small_batch_linear(row.unsqueeze(0), w, b) for row in x])
    expected = (
        uq._bf16_gemv_rows(x, w) + b
        if n == 1 or n > 256
        else torch.cat([F.linear(row.unsqueeze(0), w, b) for row in x])
    )
    assert torch.equal(single, expected)
    for m in range(2, 9):
        assert torch.equal(small_batch_linear(x[:m], w, b), single[:m])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_kill_switch_restores_legacy_wxt(monkeypatch):
    """FREETOKEN_SMALL_M_SPLITK=0 restores the legacy w @ x.T path for a one-flag A/B."""
    import freetoken.layers.quantization.linear.unquantized as uq

    g = torch.Generator(device="cuda").manual_seed(5)
    w = torch.randn(512, 2560, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    x = torch.randn(3, 2560, device="cuda", dtype=torch.bfloat16, generator=g)

    monkeypatch.setattr(uq, "_SMALL_M_DISPATCH", False)
    legacy = uq.small_batch_linear(x, w)
    expected = (w @ x.T).T.contiguous()
    assert torch.equal(legacy, expected)

    monkeypatch.setattr(uq, "_SMALL_M_DISPATCH", True)
    dispatched = uq.small_batch_linear(x, w)
    ref = x.float() @ w.float().T
    assert ((dispatched.float() - ref).norm() / ref.norm()).item() < 5e-3
