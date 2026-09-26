"""MMVQ over several input vectors: each vector's output must equal its own single-vector
launch bitwise (the kernel reads every weight block once for up to 4 vectors per block)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@requires_cuda
@pytest.mark.parametrize("batch", [2, 3, 4, 5, 9])
def test_mmvq_batched_vectors_match_single_vector_launches(batch):
    from gguf import GGMLQuantizationType
    from gguf.quants import quantize

    from freetoken.kernel.gguf import ggml_dequantize, ggml_mul_mat_vec_a8

    rows, cols = 96, 512
    qtype = int(GGMLQuantizationType.Q8_0)
    gen = np.random.default_rng(0)
    packed = quantize(gen.standard_normal((rows, cols)).astype(np.float32), qtype)
    weight = torch.from_numpy(np.ascontiguousarray(packed)).cuda()
    x = torch.randn(batch, cols, dtype=torch.bfloat16, device="cuda")

    got = ggml_mul_mat_vec_a8(weight, x, qtype, rows)
    single = torch.cat(
        [ggml_mul_mat_vec_a8(weight, x[i : i + 1], qtype, rows) for i in range(batch)]
    )
    assert torch.equal(got, single)

    ref = x.float() @ ggml_dequantize(weight, qtype, rows, cols, torch.float32).T
    torch.testing.assert_close(got.float(), ref, rtol=3e-2, atol=0.5)


# gguf-py's `quantize()` has no encoder for these formats, so the a0adc18 test above
# never covered them even though the UD-IQ4_XS checkpoint decodes through exactly these
# types. The kernel only reads bytes, so a random block with a finite fp16 scale is enough
# to exercise the same batched-vs-single-vector equality the Q8_0 test checks above.
_IQ_FORMATS = [
    ("IQ3_S", 256, 110),
    ("IQ4_XS", 256, 136),
    ("IQ4_NL", 32, 18),
    ("IQ3_XXS", 256, 98),
]


@requires_cuda
@pytest.mark.parametrize("fmt,blk_elems,blk_bytes", _IQ_FORMATS)
@pytest.mark.parametrize("batch", [2, 3, 4, 5])
def test_mmvq_batched_vectors_match_single_vector_launches_iq(fmt, blk_elems, blk_bytes, batch):
    from gguf import GGMLQuantizationType

    from freetoken.kernel.gguf import ggml_mul_mat_vec_a8

    rows = 96
    cols = blk_elems * 2
    qtype = int(getattr(GGMLQuantizationType, fmt))
    rng = np.random.default_rng(hash(fmt) & 0xFFFF)
    row_bytes = (cols // blk_elems) * blk_bytes
    raw = rng.integers(0, 256, size=(rows, row_bytes), dtype=np.uint8)
    scale_bytes = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)
    for b in range(cols // blk_elems):
        off = b * blk_bytes
        raw[:, off : off + 2] = scale_bytes
    weight = torch.from_numpy(np.ascontiguousarray(raw)).cuda()
    x = torch.randn(batch, cols, dtype=torch.bfloat16, device="cuda")

    got = ggml_mul_mat_vec_a8(weight, x, qtype, rows)
    single = torch.cat(
        [ggml_mul_mat_vec_a8(weight, x[i : i + 1], qtype, rows) for i in range(batch)]
    )
    assert torch.equal(got, single)


@requires_cuda
def test_host_resident_embedding_matches_device_lookup():
    from gguf import GGMLQuantizationType
    from gguf.quants import quantize

    from freetoken.layers.gguf import GGUFEmbedding

    vocab, dim = 300, 2560
    qtype = int(GGMLQuantizationType.Q8_0)
    packed = quantize(
        np.random.default_rng(1).standard_normal((vocab, dim)).astype(np.float32), qtype
    )
    table = torch.from_numpy(np.ascontiguousarray(packed)).cuda()

    device = GGUFEmbedding(vocab, dim, qtype)
    device.qweight = table.clone()
    host = GGUFEmbedding(vocab, dim, qtype)
    host.load_state_dict({"qweight": table.clone()})
    assert host._host_rows is not None and not host.qweight.is_cuda

    ids = torch.tensor([[0, 299, 7], [7, 150, 1]], dtype=torch.int32, device="cuda")
    assert torch.equal(host.forward(ids), device.forward(ids))
