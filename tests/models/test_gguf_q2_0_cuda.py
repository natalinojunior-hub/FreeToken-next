"""GPU tests for GGML_TYPE_Q2_0 (id 42) MMVQ and MoE-vec kernels.

Not run in this change (CPU-only environment, no CUDA build) -- follows the same
patterns as ``test_gguf_mmvq_batch.py`` (batched-vs-single MMVQ, MMVQ-vs-dequant) and
``test_moe_vec_grid_z.py`` (packing a synthetic quant bank). The dequant-vs-kernel
reference math for MoE reuses ``fused_experts_gguf``'s own dequant fallback path
(``_fused_experts_dequant``) rather than re-deriving it, since that path already covers
Q2_0 once ``GGML_Q2_0`` is in ``DEQUANT_TYPES``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from freetoken.models.gguf.dequant import GGML_Q2_0

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _raw_q2_0(rows: int, row_bytes: int, seed: int) -> np.ndarray:
    """Random Q2_0 rows with a finite fp16 scale per 18-byte block (18 = 2 + 16)."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(rows, row_bytes), dtype=np.uint8)
    scale_bytes = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
    for off in range(0, row_bytes, 18):
        raw[:, off : off + 2] = scale_bytes
    return raw


@requires_cuda
@pytest.mark.parametrize("batch", [2, 3, 4, 5, 9])
def test_mmvq_q2_0_batched_vectors_match_single_vector_launches(batch):
    from freetoken.kernel.gguf import ggml_mul_mat_vec_a8

    rows, cols = 96, 128  # 2 blocks/row
    row_bytes = (cols // 64) * 18
    raw = _raw_q2_0(rows, row_bytes, seed=0)
    weight = torch.from_numpy(np.ascontiguousarray(raw)).cuda()
    x = torch.randn(batch, cols, dtype=torch.bfloat16, device="cuda")

    got = ggml_mul_mat_vec_a8(weight, x, GGML_Q2_0, rows)
    single = torch.cat(
        [ggml_mul_mat_vec_a8(weight, x[i : i + 1], GGML_Q2_0, rows) for i in range(batch)]
    )
    assert torch.equal(got, single)


@requires_cuda
def test_mmvq_q2_0_matches_dequant_matmul():
    from freetoken.kernel.gguf import ggml_dequantize, ggml_mul_mat_vec_a8

    rows, cols = 96, 512
    row_bytes = (cols // 64) * 18
    raw = _raw_q2_0(rows, row_bytes, seed=1)
    weight = torch.from_numpy(np.ascontiguousarray(raw)).cuda()
    x = torch.randn(4, cols, dtype=torch.bfloat16, device="cuda")

    got = ggml_mul_mat_vec_a8(weight, x, GGML_Q2_0, rows)
    ref = x.float() @ ggml_dequantize(weight, GGML_Q2_0, rows, cols, torch.float32).T
    torch.testing.assert_close(got.float(), ref, rtol=3e-2, atol=0.5)


def _pack_q2_0(slots: int, out_features: int, k: int, dev) -> torch.Tensor:
    """Valid Q2_0 expert bank: per 64-element block, a finite fp16 scale then 16
    bytes of 2-bit codes. Mirrors ``_pack_q4_0`` in test_moe_vec_grid_z.py.
    """
    nb = k // 64
    codes = torch.randint(0, 256, (slots, out_features, nb, 16), dtype=torch.uint8)
    scale = (0.02 + 0.03 * torch.rand(slots, out_features, nb)).to(torch.float16)
    sb = scale.view(torch.uint8).reshape(slots, out_features, nb, 2)
    return torch.cat([sb, codes], dim=-1).reshape(slots, out_features, nb * 18).contiguous().to(dev)


@requires_cuda
def test_moe_vec_q2_0_matches_dequant_matmul_path():
    """fused_experts_gguf's MMVQ-kernel path vs its own dequant+matmul fallback path,
    both fed the same Q2_0 expert banks (forced via DEQUANT_MIN_TOKENS, like the
    kernel-vs-chunking comparison in test_moe_vec_grid_z.py)."""
    import freetoken.moe.fused_q4_0 as fq

    dev = torch.device("cuda")
    torch.manual_seed(0)
    slots, h, inter, top_k, tokens = 8, 256, 128, 2, 6

    gate_up = _pack_q2_0(slots, 2 * inter, h, dev)
    down = _pack_q2_0(slots, h, inter, dev)
    ids = torch.randint(0, slots, (tokens, top_k), dtype=torch.int32, device=dev)
    w = torch.rand(tokens, top_k, device=dev, dtype=torch.float32)
    x = (torch.randn(tokens, h, device=dev, dtype=torch.bfloat16) * 0.5).contiguous()

    def run(min_tokens: int) -> torch.Tensor:
        saved, fq.DEQUANT_MIN_TOKENS = fq.DEQUANT_MIN_TOKENS, min_tokens
        try:
            return fq.fused_experts_gguf(x, gate_up, down, w, ids, "silu", GGML_Q2_0)
        finally:
            fq.DEQUANT_MIN_TOKENS = saved

    kernel_path = run(1 << 30)  # tokens < threshold -> ggml_moe_a8_vec kernel
    dequant_path = run(0)  # tokens >= threshold -> dequant + torch matmul
    torch.testing.assert_close(kernel_path.float(), dequant_path.float(), rtol=3e-2, atol=0.5)
