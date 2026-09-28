# SPDX-License-Identifier: Apache-2.0
"""Split-K bf16 GEMV/GEMM for the tiny-M decode windows (M = 1..8).

Why this exists
---------------
The Nsight census of the native-k2 decode loop
(`notes/campaign37-decode-kernel-census.md`) shows 13.9% of all GPU kernel time --
5.20 ms of a 33.93 ms cycle -- in `cutlass_75_*` bf16 GEMMs, 298 launches per cycle. Those
are the bf16-resident dense projections: the 48 MoE router gates plus the MTP gate
(`ffn_gate_inp`, 2,621,440 B = [512, 2560] bf16 each), the hyper-connections (4 x 48
tensors, 1.172 GiB, 82% of all resident bf16), and the GDN / QSA projections.

They are not bandwidth-bound. The router gate reads 2.62 MB in 24.12 us = **109 GB/s
effective against a 620-848 GB/s DRAM ceiling** (OPT-028), i.e. 7.3x below what the memory
can deliver. Two structural causes:

* *Occupancy.* cuBLAS launches `grid=(8,1,3)` for the gate -- 24 blocks of 128 threads on a
  GPU with 84 SMs, so most SMs idle for the whole kernel. K is not split, so there is no
  further parallelism to expose at N=512, M=3.
* *Tile quantization.* `s1688gemm_bf16_128x64_tn_align1` is a real-GEMM tile shape. At M=3
  a 128-row tile is 2.3% useful, and `align1` means 1-element vectorization.

Splitting K creates the missing parallelism: the grid becomes `cdiv(N, BLOCK_N) * SPLIT_K`
programs, enough to fill every SM, while each weight byte is still read exactly once, so
total traffic is unchanged. The op stays bandwidth-bound but now near the achievable ceiling
instead of a seventh of it.

Determinism
-----------
Partials are summed in a fixed split order by a second kernel, never by atomics, so the
result is bit-reproducible run to run. It is *not* bit-identical to cuBLAS: reassociating a
length-K fp32 sum moves the last bits. That is the same class of change the already-shipped
`w @ x.T` small-batch path makes (measured ~4e-3 relative against `F.linear`, both within
~3e-3 of fp32), and a non-identical output SHA is acceptable provided the decode stays
valid.

Only for M <= 8; above that this is a real GEMM and cuBLAS/CUTLASS wins.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# tl.dot's smallest operand dimension is 16, so M is padded up to this. The padding wastes
# compute, not bandwidth -- and the op is bandwidth-bound -- so it costs nothing here.
# M_REAL carries the true row count and drives every mask.
_M_PAD = 16
_MAX_M = 8


@triton.jit
def _splitk_partial(
    x_ptr,
    w_ptr,
    part_ptr,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_pp,
    stride_pm,
    M: tl.constexpr,
    M_REAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """One (N-block, K-split) partial: part[split, m, n] = sum_k w[n,k] * x[m,k]."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_off < N
    m_off = tl.arange(0, M)
    m_mask = m_off < M_REAL

    k_per = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per
    k_end = tl.minimum(k_start + k_per, K)

    acc = tl.zeros((M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_start, k_end, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_off < k_end
        # w is [N, K] row-major: BLOCK_N rows x BLOCK_K contiguous columns.
        w_t = tl.load(
            w_ptr + n_off[:, None] * stride_wn + k_off[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # x is [M_REAL, K] row-major; load the [BLOCK_K, M] transpose tile for tl.dot.
        # Padding rows must be masked to zero: they are out of bounds, and tl.dot would
        # otherwise multiply garbage into the (discarded) padded output rows.
        x_t = tl.load(
            x_ptr + m_off[None, :] * stride_xm + k_off[:, None],
            mask=m_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        acc += tl.trans(tl.dot(w_t, x_t))

    base = part_ptr + pid_k * stride_pp + m_off[:, None] * stride_pm + n_off[None, :]
    tl.store(base, acc, mask=m_mask[:, None] & n_mask[None, :])


@triton.jit
def _splitk_reduce(
    part_ptr,
    out_ptr,
    N,
    stride_pp,
    stride_pm,
    stride_om,
    M: tl.constexpr,
    M_REAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Sum the SPLIT_K partials in fixed order and cast to the output dtype."""
    pid_n = tl.program_id(0)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_off < N
    m_off = tl.arange(0, M)
    m_mask = m_off < M_REAL

    acc = tl.zeros((M, BLOCK_N), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(
            part_ptr + s * stride_pp + m_off[:, None] * stride_pm + n_off[None, :],
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
    tl.store(
        out_ptr + m_off[:, None] * stride_om + n_off[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def _pick_config(n: int, k: int) -> tuple[int, int, int, int]:
    """(BLOCK_N, BLOCK_K, SPLIT_K, num_warps) sized to fill the GPU without over-splitting.

    Aim for ~3x the SM count in programs (252 on an RTX 5080's 84 SMs) so the tail is not a
    single partial wave, while keeping each program's K chunk at least one BLOCK_K.
    """
    block_n = 64 if n >= 256 else max(16, triton.next_power_of_2(n))
    n_blocks = triton.cdiv(n, block_n)
    split_k = max(1, min(32, triton.next_power_of_2(triton.cdiv(252, max(n_blocks, 1)))))
    block_k = 64
    while block_k > 16 and triton.cdiv(k, split_k) < block_k:
        block_k //= 2
    return block_n, block_k, split_k, 4


def small_m_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
    """``x @ w.T`` for x of 1..8 rows, split-K for occupancy. Drop-in for ``F.linear``."""
    if x.dim() != 2:
        return torch.nn.functional.linear(x, w, b)
    m, k = x.shape
    if not 1 <= m <= _MAX_M:
        return torch.nn.functional.linear(x, w, b)
    if not (x.is_cuda and w.is_cuda and x.dtype == w.dtype == torch.bfloat16):
        return torch.nn.functional.linear(x, w, b)
    n = w.shape[0]
    if k % 16 != 0 or n % 16 != 0 or w.stride(1) != 1 or x.stride(1) != 1:
        return torch.nn.functional.linear(x, w, b)

    block_n, block_k, split_k, warps = _pick_config(n, k)
    if split_k <= 1:
        return torch.nn.functional.linear(x, w, b)

    part = torch.empty((split_k, _M_PAD, n), dtype=torch.float32, device=x.device)
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(n, block_n), split_k)
    _splitk_partial[grid](
        x,
        w,
        part,
        n,
        k,
        x.stride(0),
        w.stride(0),
        part.stride(0),
        part.stride(1),
        M=_M_PAD,
        M_REAL=m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
        num_warps=warps,
    )
    _splitk_reduce[(triton.cdiv(n, block_n),)](
        part,
        out,
        n,
        part.stride(0),
        part.stride(1),
        out.stride(0),
        M=_M_PAD,
        M_REAL=m,
        BLOCK_N=block_n,
        SPLIT_K=split_k,
        num_warps=warps,
    )
    return out + b if b is not None else out


__all__ = ["small_m_linear"]
