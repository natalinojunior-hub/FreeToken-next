import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _dequant(packed, scale, glob):
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=packed.device
    )
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(*packed.shape[:-1], -1)
    vals = lut[codes.long()].reshape(*codes.shape[:-1], -1, 16)
    return (vals * scale.float().unsqueeze(-1)).flatten(-2) * glob.float().unsqueeze(-1)


@pytest.mark.parametrize("magnitude", [1.0, 0.02, 3e-4])
def test_quantize_nvfp4_roundtrip_and_decode_kernel(magnitude):
    """BF16 -> NVFP4 pieces dequantize close to the source, and the production decode GEMV over
    a packed bank reproduces x @ dequant(w)^T -- nibble order, block scale and global agree."""
    from freetoken.models.nvfp4_banks import quantize_nvfp4
    from freetoken.moe.fused_nvfp4 import _decode_gemm_marlin

    torch.manual_seed(0)
    E, N, K = 3, 256, 512
    w = (torch.randn(E, N, K, device="cuda") * magnitude).to(torch.bfloat16)
    packed, scale, glob = quantize_nvfp4(w)
    assert packed.shape == (E, N, K // 2) and scale.shape == (E, N, K // 16)
    deq = _dequant(packed, scale, glob)
    rel = (deq - w.float()).norm() / w.float().norm()
    assert rel < 0.12, rel

    a = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[2, 0]], device="cuda", dtype=torch.int32)
    out = torch.empty(1, 2, N, device="cuda", dtype=torch.bfloat16)
    g_rows = glob.expand(E, N).contiguous()
    _decode_gemm_marlin(
        a, packed, scale, g_rows, out, torch.ones(1, 2, device="cuda"), ids, False, False
    )
    ref = torch.stack([a.float() @ deq[e].t() for e in (2, 0)], dim=1)
    err = (out.float() - ref).norm() / ref.norm()
    assert err < 1e-2, err
