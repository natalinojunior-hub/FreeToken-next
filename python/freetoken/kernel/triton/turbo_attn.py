"""Tile readers that let a Triton attention kernel consume turbo-coded KV directly.

Per tile the work reduces to: gather the packed byte(s) each element lives in, look the index up in
the centroid book, multiply by that (token, head, group) norm. No de-rotation, because the values
are *stored* rotated and the caller pre-rotates Q -- rotate is orthogonal, so
``(W q) . (W k)^T == q . k^T`` -- and V accumulates rotated as well, undone once per output row.
That is what keeps this path off the materialize-everything cliff the reference measured at
-10.8 % TG at 16K and -28.7 % at 64K.

Layout mirrors ``kvcache/turbo_pool.py``: codes ``[tokens, heads, GROUPS * CODE_BYTES]`` uint8 and
norms ``[tokens, heads, GROUPS]`` fp16, one storage layer sliced per call. K comes back ``[D, N]``
for the score dot, V ``[N, D]`` for the value dot, matching the tile shapes the kernels already use.
"""

from __future__ import annotations

import triton
import triton.language as tl

# ``BOOK`` constexpr of the tile readers; ``cent`` is unused by fp8.
BOOK_TURBO4 = tl.constexpr(0)
BOOK_TURBO3 = tl.constexpr(1)
BOOK_FP8 = tl.constexpr(2)
BOOK_CODE = {"turbo4": 0, "turbo3": 1, "fp8": 2}


@triton.jit
def turbo_k_tile(
    codes_ptr,  # uint8 [tokens, heads, GROUPS * CODE_BYTES]
    norm_ptr,  # fp16  [tokens, heads, GROUPS]
    cent_ptr,  # fp32  [8] or [16]
    slots,  #       [N]   token rows in this tile
    kv_head,
    stride_ct,
    stride_ch,
    stride_nt,
    stride_nh,
    offs_d,  # int32 [D] element index inside one head row
    mask_n,  # bool  [N]
    BOOK: tl.constexpr,
    out_dtype: tl.constexpr,
):
    """K tile [D, N], rotated domain: ``centroid[idx] * norm``."""
    col = slots[None, :] * stride_ct + kv_head * stride_ch
    ncol = slots[None, :] * stride_nt + kv_head * stride_nh
    grp = offs_d // 128
    jj = offs_d % 128
    if BOOK == BOOK_FP8:
        raw = tl.load(codes_ptr + col + (grp * 128 + jj)[:, None], mask=mask_n[None, :], other=0)
        return raw.to(tl.float8e4nv, bitcast=True).to(out_dtype)  # fp8 norm is always 1
    nrm = tl.load(norm_ptr + ncol + grp[:, None], mask=mask_n[None, :], other=0.0).to(tl.float32)
    if BOOK == BOOK_TURBO3:
        low = tl.load(
            codes_ptr + col + (grp * 48 + jj // 4)[:, None], mask=mask_n[None, :], other=0
        )
        bit = tl.load(
            codes_ptr + col + (grp * 48 + 32 + jj // 8)[:, None], mask=mask_n[None, :], other=0
        )
        idx = ((low.to(tl.int32) >> ((jj % 4) * 2)[:, None]) & 3) | (
            ((bit.to(tl.int32) >> (jj % 8)[:, None]) & 1) << 2
        )
    else:
        byte = tl.load(
            codes_ptr + col + (grp * 64 + jj // 2)[:, None], mask=mask_n[None, :], other=0
        )
        idx = (byte.to(tl.int32) >> ((jj % 2) * 4)[:, None]) & 15
    vals = tl.load(cent_ptr + idx)
    return (vals * nrm).to(out_dtype)


@triton.jit
def turbo_v_tile(
    codes_ptr,
    norm_ptr,
    cent_ptr,
    slots,
    kv_head,
    stride_ct,
    stride_ch,
    stride_nt,
    stride_nh,
    offs_d,  # int32 [DV]
    mask_n,  # bool  [N]
    BOOK: tl.constexpr,
    out_dtype: tl.constexpr,
):
    """V tile [N, D], rotated domain, so ``tl.dot(p, v)`` needs no transpose."""
    row = slots[:, None] * stride_ct + kv_head * stride_ch
    nrow = slots[:, None] * stride_nt + kv_head * stride_nh
    grp = offs_d // 128
    jj = offs_d % 128
    if BOOK == BOOK_FP8:
        raw = tl.load(codes_ptr + row + (grp * 128 + jj)[None, :], mask=mask_n[:, None], other=0)
        return raw.to(tl.float8e4nv, bitcast=True).to(out_dtype)  # fp8 norm is always 1
    nrm = tl.load(norm_ptr + nrow + grp[None, :], mask=mask_n[:, None], other=0.0).to(tl.float32)
    if BOOK == BOOK_TURBO3:
        low = tl.load(
            codes_ptr + row + (grp * 48 + jj // 4)[None, :], mask=mask_n[:, None], other=0
        )
        bit = tl.load(
            codes_ptr + row + (grp * 48 + 32 + jj // 8)[None, :], mask=mask_n[:, None], other=0
        )
        idx = ((low.to(tl.int32) >> ((jj % 4) * 2)[None, :]) & 3) | (
            ((bit.to(tl.int32) >> (jj % 8)[None, :]) & 1) << 2
        )
    else:
        byte = tl.load(
            codes_ptr + row + (grp * 64 + jj // 2)[None, :], mask=mask_n[:, None], other=0
        )
        idx = (byte.to(tl.int32) >> ((jj % 2) * 4)[None, :]) & 15
    vals = tl.load(cent_ptr + idx)
    return (vals * nrm).to(out_dtype)


@triton.jit
def _dequant_rows_kernel(
    codes_ptr,
    norm_ptr,
    cent_ptr,
    slots_ptr,
    out_ptr,
    n,
    stride_ct,
    stride_ch,
    stride_nt,
    stride_nh,
    stride_ot,
    stride_oh,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BOOK: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    head = tl.program_id(1)
    mask_n = offs_n < n
    slots = tl.load(slots_ptr + offs_n, mask=mask_n, other=0).to(tl.int64)
    offs_d = tl.arange(0, D)
    rows = turbo_v_tile(
        codes_ptr,
        norm_ptr,
        cent_ptr,
        slots,
        head,
        stride_ct,
        stride_ch,
        stride_nt,
        stride_nh,
        offs_d,
        mask_n,
        BOOK,
        tl.bfloat16,
    )
    dst = out_ptr + offs_n[:, None].to(tl.int64) * stride_ot + head * stride_oh + offs_d[None, :]
    tl.store(dst, rows, mask=mask_n[:, None])


def dequant_rows(codes, norm, cent, slots, book: int, out) -> None:
    """``out[i] = decode(codes[slots[i]])`` for every kv head, bf16, in the slab's (rotated)
    domain: the KV rows of a paged prefix gathered dense for a bf16 attention kernel."""
    n, heads, dim = slots.numel(), out.shape[1], out.shape[2]
    if n == 0:
        return
    block_n = 32
    _dequant_rows_kernel[(triton.cdiv(n, block_n), heads)](
        codes,
        norm,
        cent,
        slots,
        out,
        n,
        codes.stride(0),
        codes.stride(1),
        norm.stride(0),
        norm.stride(1),
        out.stride(0),
        out.stride(1),
        D=dim,
        BLOCK_N=block_n,
        BOOK=book,
    )
