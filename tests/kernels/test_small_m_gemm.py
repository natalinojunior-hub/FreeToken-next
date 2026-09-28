"""Split-K bf16 small-M GEMV (`small_m_linear`) against cuBLAS and an fp32 reference.

The kernel attacks the `cutlass_75_*` bf16 GEMM family (13.9% of decode GPU time,
5.20 ms/cycle) whose defect is occupancy, not bandwidth: the router gate launches 24 blocks
on 84 SMs and reads 2.62 MB at 109 GB/s against a 620-848 GB/s ceiling. Splitting K raises
the program count without changing traffic.

Reference comparison is fp32 on purpose: the split-K reduce reassociates the length-K sum, so
bit-equality with cuBLAS is neither expected nor wanted. The shipped `w @ x.T` small-batch path
measures ~4e-3 relative against `F.linear` (both ~3e-3 from fp32); this kernel must land in the
same class. Shapes below are the real resident-bf16 decode operands: the router gate
`ffn_gate_inp` [512, 2560], the fused hyper-connection down+inject block [336, 10240], the
hyper-connection up [10240, 320], and a square [2560, 2560] control.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from freetoken.kernel.triton.small_m_gemm import small_m_linear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# (name, N=out_features, K=in_features). All native-bf16 resident decode operands.
SHAPES = [
    ("router_gate", 512, 2560),
    ("hc_down_inject", 336, 10240),
    ("hc_up", 10240, 320),
    ("square", 2560, 2560),
]
MS = (1, 2, 3, 4, 8)

# The split-K reduce reassociates in fp32 then casts to bf16, so it tracks the fp32 reference
# at least as tightly as cuBLAS does. Measured: rel-to-fp32 ~1.6-1.8e-3, rel-to-cuBLAS <=2.8e-3.
REL_FP32_MAX = 5e-3
REL_CUBLAS_MAX = 6e-3


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _make(n, k, m, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    return x, w


@pytest.mark.parametrize("name,n,k", SHAPES, ids=[s[0] for s in SHAPES])
@pytest.mark.parametrize("m", MS)
def test_matches_fp32_and_cublas(name, n, k, m):
    x, w = _make(n, k, m)
    out = small_m_linear(x, w)
    cublas = F.linear(x, w)
    ref = x.float() @ w.float().T

    assert out.shape == cublas.shape == (m, n)
    assert out.dtype == torch.bfloat16
    # The kernel must be a faithful bf16 GEMV: close to the fp32 truth, and no further from
    # cuBLAS than the ~4e-3 the shipped w @ x.T path already costs.
    assert _rel(out, ref) < REL_FP32_MAX
    assert _rel(cublas, ref) < REL_FP32_MAX
    assert _rel(out, cublas) < REL_CUBLAS_MAX


@pytest.mark.parametrize("name,n,k", SHAPES, ids=[s[0] for s in SHAPES])
@pytest.mark.parametrize("m", MS)
def test_deterministic_bitwise(name, n, k, m):
    """Fixed split-order reduce, no atomics: two calls on the same inputs are bit-identical."""
    x, w = _make(n, k, m)
    a = small_m_linear(x, w)
    b = small_m_linear(x, w)
    assert torch.equal(a, b)


@pytest.mark.parametrize("name,n,k", SHAPES, ids=[s[0] for s in SHAPES])
@pytest.mark.parametrize("m", (2, 3, 8))
def test_bias_matches_cublas(name, n, k, m):
    x, w = _make(n, k, m)
    g = torch.Generator(device="cuda").manual_seed(7)
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02
    out = small_m_linear(x, w, b)
    ref = F.linear(x, w, b).float()
    assert out.shape == (m, n)
    assert _rel(out, ref) < REL_CUBLAS_MAX


# --- guard rails: everything outside the split-K envelope must fall through to F.linear exactly ---


@pytest.mark.parametrize(
    "m",
    [0, 9, 16, 33],
    ids=lambda v: f"M{v}",
)
def test_m_out_of_range_falls_back(m):
    """M outside 1..8 is a real GEMM: cuBLAS wins, so fall through bit-exactly."""
    k, n = 2560, 512
    if m == 0:
        x = torch.empty(0, k, device="cuda", dtype=torch.bfloat16)
    else:
        x, _ = _make(n, k, m)
    _, w = _make(n, k, 1)
    w = w[:n]
    assert torch.equal(small_m_linear(x, w), F.linear(x, w))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_non_bf16_falls_back(dtype):
    x = torch.randn(3, 2560, device="cuda", dtype=dtype)
    w = torch.randn(512, 2560, device="cuda", dtype=dtype)
    assert torch.equal(small_m_linear(x, w), F.linear(x, w))


def test_cpu_falls_back():
    x = torch.randn(3, 2560, dtype=torch.bfloat16)
    w = torch.randn(512, 2560, dtype=torch.bfloat16)
    assert torch.equal(small_m_linear(x, w), F.linear(x, w))


def test_non_unit_inner_stride_falls_back():
    """A column-major activation (stride(1) != 1) is out of contract; fall through bit-exactly."""
    x = torch.randn(2560, 3, device="cuda", dtype=torch.bfloat16).t()  # [3, 2560], stride (1, 3)
    w = torch.randn(512, 2560, device="cuda", dtype=torch.bfloat16)
    assert x.stride(1) != 1
    assert torch.equal(small_m_linear(x, w), F.linear(x, w))


@pytest.mark.parametrize("n,k", [(512, 2568), (520, 2560)], ids=["oddK", "oddN"])
def test_unaligned_falls_back(n, k):
    """N or K not a multiple of 16 breaks tl.dot's 16-lane floor; fall through bit-exactly."""
    x = torch.randn(3, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(small_m_linear(x, w), F.linear(x, w))


def test_large_n_no_split_falls_back():
    """When N alone fills the GPU, split_k resolves to 1 and the kernel declines (bit-exact)."""
    n, k, m = 16384, 2560, 3
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    from freetoken.kernel.triton.small_m_gemm import _pick_config

    assert _pick_config(n, k)[2] == 1
    assert torch.equal(small_m_linear(x, w), F.linear(x, w))


def test_3d_activation_falls_back():
    """A [B, T, K] activation is not the decode window; fall through to F.linear."""
    x = torch.randn(2, 4, 2560, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(512, 2560, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(small_m_linear(x, w), F.linear(x, w))
