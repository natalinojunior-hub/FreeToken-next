"""Bit-exactness for the GGUF quantize-once path (GGUFMergedLinear decode).

The quantize-once hoist (``layers/gguf.py`` ``GGUFMergedLinear.forward``) quantizes
the shared activation ``x`` -> q8_1 once and drives every MMVQ part through
``ggml_mul_mat_vec_a8_prequant``. That must be BIT-identical to the original
per-part ``ggml_mul_mat_vec_a8`` (which re-quantizes ``x`` internally), because
``quantize_row_q8_1`` is a deterministic pure function of ``x`` and the MMVQ kernel
receives the exact same ``quant_X`` bits. These tests assert raw-bit equality on the
real CUDA kernels (raw bits, so NaN/Inf compare equal when the bits match).
"""

from __future__ import annotations

import pytest
import torch

from freetoken.models.gguf.dequant import (
    BLOCK_SHAPE,
    GGML_IQ3_XXS,
    GGML_Q4_K,
    GGML_Q6_K,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="quantize-once bit-exactness needs the compiled CUDA kernels",
)

_VIEW = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(_VIEW[t.dtype])


def _qweight(out_features: int, in_features: int, qtype: int, device: str) -> torch.Tensor:
    block, type_size = BLOCK_SHAPE[qtype]
    row_bytes = in_features // block * type_size
    return torch.randint(0, 256, (out_features, row_bytes), dtype=torch.uint8, device=device)


@requires_cuda
@pytest.mark.parametrize("qtype", [GGML_Q4_K, GGML_Q6_K, GGML_IQ3_XXS])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_prequant_split_bit_exact(qtype: int, dtype: torch.dtype) -> None:
    """quantize_row_q8_1 + prequant == ggml_mul_mat_vec_a8, bit-for-bit."""
    from freetoken.kernel.gguf import (
        ggml_mul_mat_vec_a8,
        ggml_mul_mat_vec_a8_prequant,
        ggml_quantize_row_q8_1,
    )

    torch.manual_seed(0)
    row, col, vecs = 512, 256, 1
    x = torch.randn(vecs, col, dtype=dtype, device="cuda")
    w = _qweight(row, col, qtype, "cuda")

    ref = ggml_mul_mat_vec_a8(w, x, qtype, row)
    qx = ggml_quantize_row_q8_1(x)
    got = ggml_mul_mat_vec_a8_prequant(w, qx, qtype, row, col, vecs, dtype)

    assert torch.equal(_bits(ref), _bits(got))


@requires_cuda
@pytest.mark.parametrize(
    "output_sizes,qtypes",
    [
        ([512, 256], [GGML_Q4_K, GGML_IQ3_XXS]),
        ([256, 256, 256], [GGML_Q4_K, GGML_Q6_K, GGML_IQ3_XXS]),
    ],
)
def test_merged_linear_forward_bit_exact(output_sizes: list[int], qtypes: list[int]) -> None:
    """GGUFMergedLinear.forward (quantize-once) == per-part reference, bit-for-bit."""
    from freetoken.kernel.gguf import ggml_mul_mat_vec_a8
    from freetoken.layers.gguf import GGUFMergedLinear

    torch.manual_seed(0)
    in_features = 256
    dtype = torch.bfloat16
    lin = GGUFMergedLinear(in_features, output_sizes, qtypes, has_bias=True)
    for name, qt, osz in zip(lin.part_names, qtypes, output_sizes):
        setattr(lin, name, _qweight(osz, in_features, qt, "cuda"))
    lin.bias = torch.randn(lin.out_features, dtype=dtype, device="cuda")

    x = torch.randn(1, in_features, dtype=dtype, device="cuda")
    got = lin.forward(x)

    ref = (
        torch.cat(
            [
                ggml_mul_mat_vec_a8(getattr(lin, name), x, qt, osz)
                for name, qt, osz in zip(lin.part_names, qtypes, output_sizes)
            ],
            dim=-1,
        )
        + lin.bias
    )

    assert got.shape == ref.shape
    assert torch.equal(_bits(got), _bits(ref))
