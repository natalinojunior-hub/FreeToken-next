import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="fp8 tensor cores need sm_89+",
)


def test_rowwise_quant_roundtrip_runtime_strides():
    from freetoken.kernel.triton.fp8_pertensor_linear import rowwise_quant_fp8

    x = torch.randn(300, 2 * 1040, device="cuda", dtype=torch.bfloat16)[:, ::2]  # strided rows
    x[7] *= 1000.0  # per-row scale: one hot row must not flatten the others
    q, s = rowwise_quant_fp8(x)
    back = q.float() * s
    rel = (back - x.float()).abs().amax(1) / x.float().abs().amax(1)
    assert rel.max().item() < 0.07  # e4m3: 3 mantissa bits
    assert torch.equal(s[:, 0], x.float().abs().amax(1) / 448.0)


def test_fp8_rowwise_matmul_close_to_bf16():
    from freetoken.kernel.triton.fp8_pertensor_linear import fp8_rowwise_matmul

    torch.manual_seed(0)
    x = torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16) * 0.02
    got, want = fp8_rowwise_matmul(x, w).float(), (x @ w.T).float()
    cos = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim=0).item()
    assert cos > 0.999, cos
