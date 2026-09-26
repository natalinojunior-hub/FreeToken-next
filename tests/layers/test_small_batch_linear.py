import torch

from freetoken.layers.quantization.linear.unquantized import small_batch_linear


def test_small_batch_linear_is_contiguous_and_matches():
    """2..8 rows take the w @ x.T GEMV route; consumers (hc_silu) need row-major output."""
    w = torch.randn(48, 64, dtype=torch.bfloat16)
    for rows in (1, 2, 5, 8, 9):
        x = torch.randn(rows, 64, dtype=torch.bfloat16)
        y = small_batch_linear(x, w)
        assert y.is_contiguous()
        torch.testing.assert_close(y.float(), (x @ w.T).float(), rtol=2e-2, atol=2e-2)
