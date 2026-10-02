"""MTP verify rows on the NVFP4 GEMV must be bit-identical to RAW M==1 decode."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("n", [256, 4096, 151936], ids=["splitk-small-n", "splitk", "single-pass"])
@pytest.mark.parametrize("transposed", [False, True])
def test_row_exact_gemv_rows_equal_m1(monkeypatch, n, transposed):
    from freetoken.kernel.triton.nvfp4_linear import (
        FP8,
        nvfp4_dense_linear,
        nvfp4_dense_linear_t,
        nvfp4_transpose_resident,
    )

    torch.manual_seed(3)
    k = 2048
    pk = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    sc = (torch.rand(n, k // 16, device="cuda") * 2 + 0.25).to(FP8)
    g = (torch.rand(n, device="cuda") * 0.01 + 0.001).to(torch.float16)
    linear = nvfp4_dense_linear
    if transposed:
        wt, st = nvfp4_transpose_resident(pk, sc)
        pk, sc, linear = wt, st, nvfp4_dense_linear_t
    monkeypatch.setenv("FREETOKEN_ROW_INVARIANT_LINEAR", "1")
    for rows in (2, 3, 6, 8):
        x = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda")
        got = linear(x, pk, sc, g)
        want = torch.cat([linear(x[i : i + 1], pk, sc, g) for i in range(rows)])
        assert torch.equal(got, want)


@pytest.mark.parametrize("input_scale", [None, 0.05], ids=["w8a16", "w8a8"])
def test_row_exact_fp8_pertensor_rows_equal_m1(monkeypatch, input_scale):
    from freetoken.kernel.triton.fp8_pertensor_linear import fp8_pertensor_linear

    torch.manual_seed(4)
    n, k = 1536, 2048
    w = (torch.randn(n, k, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    ws = torch.rand(n, device="cuda") * 0.01 + 0.001
    scale = None if input_scale is None else torch.tensor(input_scale, device="cuda")
    monkeypatch.setenv("FREETOKEN_ROW_INVARIANT_LINEAR", "1")
    for rows in (2, 5, 8):
        x = torch.randn(rows, k, dtype=torch.bfloat16, device="cuda")
        got = fp8_pertensor_linear(x, w, ws, input_scale=scale, uniform_scale=True)
        want = torch.cat(
            [
                fp8_pertensor_linear(x[i : i + 1], w, ws, input_scale=scale, uniform_scale=True)
                for i in range(rows)
            ]
        )
        assert torch.equal(got, want)
