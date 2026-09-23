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
