# SPDX-License-Identifier: Apache-2.0
"""KV RAM tier kernels: stores and page gathers across the device slab and host tier.

The host tier is pinned host memory addressed zero-copy by the GPU (UVA), so every copy here is
stream-ordered with the attention kernels that read it: no host synchronization, no events, and
fixed addresses under CUDA graph capture.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _tiered_store_kernel(
    k_ptr,
    v_ptr,
    loc_ptr,
    k_dev_ptr,
    v_dev_ptr,
    k_host_ptr,
    v_host_ptr,
    stride_src,
    device_rows,
    total_rows,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    token = tl.program_id(0).to(tl.int64)
    slot = tl.load(loc_ptr + token).to(tl.int64)
    in_range = (slot >= 0) & (slot < total_rows)
    on_device = in_range & (slot < device_rows)
    on_host = in_range & (slot >= device_rows)
    row = tl.where(on_device, slot, slot - device_rows)
    row = tl.maximum(row, 0)
    for start in range(0, ROW, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < ROW
        k = tl.load(k_ptr + token * stride_src + cols, mask=mask)
        v = tl.load(v_ptr + token * stride_src + cols, mask=mask)
        tl.store(k_dev_ptr + row * ROW + cols, k, mask=mask & on_device)
        tl.store(v_dev_ptr + row * ROW + cols, v, mask=mask & on_device)
        # The RAM tier may be narrower (FP8): the store casts to its element type.
        tl.store(
            k_host_ptr + row * ROW + cols,
            k.to(k_host_ptr.dtype.element_ty),
            mask=mask & on_host,
        )
        tl.store(
            v_host_ptr + row * ROW + cols,
            v.to(v_host_ptr.dtype.element_ty),
            mask=mask & on_host,
        )


def tiered_store_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    out_loc: torch.Tensor,
    device_kv: tuple[torch.Tensor, torch.Tensor],
    host_kv: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Scatter ``k``/``v`` rows to token slots; slots past the device slab land in the host tier.

    ``device_kv``/``host_kv`` are one layer's contiguous ``[pages, page_size, heads, dim]`` slabs.
    Slots outside both tiers are dropped (the kernels never read them).
    """
    k_dev, v_dev = device_kv
    k_host, v_host = host_kv
    row = k_dev[0].numel() // k_dev.shape[1]
    tokens = out_loc.shape[0]
    for t in (k_dev, v_dev, k_host, v_host):
        if not t.is_contiguous():
            raise ValueError("tiered KV slabs must be contiguous")
    if k_dev.dtype != k.dtype or v_dev.dtype != k.dtype or k_host.dtype != v_host.dtype:
        raise ValueError("device slabs must match the K/V dtype; RAM K/V must share one dtype")
    k = k.reshape(tokens, row)
    v = v.reshape(tokens, row)
    if k.stride(1) != 1 or v.stride() != k.stride():
        raise ValueError("tiered KV store needs row-contiguous K/V inputs")
    if not tokens:
        return
    device_rows = k_dev.shape[0] * k_dev.shape[1]
    total_rows = device_rows + k_host.shape[0] * k_host.shape[1]
    _tiered_store_kernel[(tokens,)](
        k,
        v,
        out_loc,
        k_dev,
        v_dev,
        k_host,
        v_host,
        k.stride(0),
        device_rows,
        total_rows,
        ROW=row,
        BLOCK=min(triton.next_power_of_2(row), 1024),
        num_warps=4,
    )


@triton.jit
def _page_gather_kernel(
    src_ptr,
    dst_ptr,
    pages_ptr,
    PAGE_ELEMS: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    i = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    page = tl.load(pages_ptr + i).to(tl.int64)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = cols < PAGE_ELEMS
    tl.store(
        dst_ptr + i * PAGE_ELEMS + cols,
        tl.load(src_ptr + page * PAGE_ELEMS + cols, mask=mask).to(dst_ptr.dtype.element_ty),
        mask=mask,
    )


def gather_pages(src: torch.Tensor, pages: torch.Tensor, dst: torch.Tensor) -> None:
    """``dst[i] = src[pages[i]]`` for contiguous page slabs; ``src`` may be pinned host memory."""
    if not (src.is_contiguous() and dst.is_contiguous()) or src.shape[1:] != dst.shape[1:]:
        raise ValueError("page gather needs contiguous slabs with one page layout")
    if pages.dtype not in (torch.int32, torch.int64) or pages.shape[0] > dst.shape[0]:
        raise ValueError("page gather index must be int32/int64 and fit the destination")
    if not pages.shape[0]:
        return
    elems = src[0].numel()
    block = 4096
    _page_gather_kernel[(pages.shape[0], triton.cdiv(elems, block))](
        src, dst, pages, PAGE_ELEMS=elems, BLOCK=block, num_warps=4
    )


@triton.jit
def _zero_kernel(dst_ptr, n, BLOCK: tl.constexpr) -> None:
    cols = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    tl.store(dst_ptr + cols, tl.zeros((BLOCK,), dtype=dst_ptr.dtype.element_ty), mask=cols < n)


def zero_tier(dst: torch.Tensor) -> None:
    """Stream-ordered zero of a (possibly host-tier) contiguous tensor.

    A CPU ``zero_()`` on the RAM tier would race kernels still queued against it."""
    if not dst.is_contiguous():
        raise ValueError("zero_tier needs a contiguous tensor")
    n = dst.numel()
    if n:
        block = 8192
        _zero_kernel[(triton.cdiv(n, block),)](dst, n, BLOCK=block, num_warps=4)


@triton.jit
def _swap_pages_kernel(
    a_ptr,
    b_ptr,
    a_idx_ptr,
    b_idx_ptr,
    active_ptr,
    stride_ag,
    stride_ap,
    stride_bg,
    stride_bp,
    PAGE_ELEMS: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    pair = tl.program_id(0)
    group = tl.program_id(1).to(tl.int64)
    chunk = tl.program_id(2)
    if tl.load(active_ptr + pair) != 0:
        cols = chunk * BLOCK + tl.arange(0, BLOCK)
        mask = cols < PAGE_ELEMS
        a = a_ptr + group * stride_ag + tl.load(a_idx_ptr + pair).to(tl.int64) * stride_ap + cols
        b = b_ptr + group * stride_bg + tl.load(b_idx_ptr + pair).to(tl.int64) * stride_bp + cols
        x = tl.load(a, mask=mask)
        y = tl.load(b, mask=mask)
        tl.store(a, y.to(a_ptr.dtype.element_ty), mask=mask)
        tl.store(b, x.to(b_ptr.dtype.element_ty), mask=mask)


def swap_pages(
    a: torch.Tensor,
    b: torch.Tensor,
    a_idx: torch.Tensor,
    b_idx: torch.Tensor,
    active: torch.Tensor,
) -> None:
    """Swap page ``a[g, a_idx[i]]`` with ``b[g, b_idx[i]]`` for every group ``g`` where
    ``active[i]``; ``a``/``b`` are ``[groups, pages, ...]`` views whose pages are contiguous.

    Pairs must not alias one another (distinct pages on each side, and no page on both)."""
    if a.shape[0] != b.shape[0] or a.shape[2:] != b.shape[2:]:
        raise ValueError("swap_pages needs matching [groups, pages, ...] layouts")
    elems = a[0, 0].numel()
    for t in (a, b):
        if t.stride(1) < elems or not t[0, 0].is_contiguous():
            raise ValueError("swap_pages needs page-contiguous slabs")
    if not a_idx.shape[0]:
        return
    block = min(triton.next_power_of_2(elems), 4096)
    _swap_pages_kernel[(a_idx.shape[0], a.shape[0], triton.cdiv(elems, block))](
        a,
        b,
        a_idx,
        b_idx,
        active,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        PAGE_ELEMS=elems,
        BLOCK=block,
        num_warps=4,
    )


@triton.jit
def _scatter_rows_kernel(
    src_ptr,
    loc_ptr,
    dst_ptr,
    stride_src,
    base_row,
    dst_rows,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    token = tl.program_id(0).to(tl.int64)
    row = tl.load(loc_ptr + token).to(tl.int64) - base_row
    if (row >= 0) & (row < dst_rows):
        for start in range(0, ROW, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            mask = cols < ROW
            x = tl.load(src_ptr + token * stride_src + cols, mask=mask)
            tl.store(dst_ptr + row * ROW + cols, x, mask=mask)


def scatter_rows(src: torch.Tensor, loc: torch.Tensor, dst: torch.Tensor, base_row: int) -> None:
    """``dst[loc[i] - base_row] = src[i]`` for rows landing inside ``dst``; others are dropped.

    ``src`` is ``[T, ...]`` with contiguous rows, ``dst`` ``[N, ...]`` contiguous and the same
    row layout; any dtype (turbo codes, norms)."""
    tokens = loc.shape[0]
    row = dst[0].numel()
    src = src.reshape(tokens, row)
    if src.stride(1) != 1 or not dst.is_contiguous() or src.dtype != dst.dtype:
        raise ValueError("scatter_rows needs contiguous rows of one dtype")
    if tokens:
        _scatter_rows_kernel[(tokens,)](
            src, loc, dst, src.stride(0), base_row, dst.shape[0],
            ROW=row, BLOCK=min(triton.next_power_of_2(row), 1024), num_warps=4,
        )  # fmt: skip


@triton.jit
def _turbo_pages_kernel(
    codes_ptr,
    norm_ptr,
    cent_ptr,
    rot_ptr,
    src_ptr,
    dst_ptr,
    out_ptr,
    stride_c_token,
    stride_c_head,
    stride_n_token,
    stride_n_head,
    stride_o_page,
    stride_o_token,
    stride_o_head,
    PAGE_SIZE: tl.constexpr,
    GROUPS: tl.constexpr,
    BOOK3: tl.constexpr,
) -> None:
    entry = tl.program_id(0)
    head = tl.program_id(1)
    page = tl.load(src_ptr + entry).to(tl.int64)
    if page >= 0:
        dst = tl.load(dst_ptr + entry).to(tl.int64)
        tok = tl.arange(0, PAGE_SIZE)
        dim = tl.arange(0, 128)
        slots = page * PAGE_SIZE + tok
        rot = tl.load(rot_ptr + dim[:, None] * 128 + dim[None, :])
        for g in range(GROUPS):
            base = codes_ptr + slots[:, None] * stride_c_token + head * stride_c_head
            if BOOK3:
                low = tl.load(base + (g * 48 + dim // 4)[None, :])
                bit = tl.load(base + (g * 48 + 32 + dim // 8)[None, :])
                idx = ((low.to(tl.int32) >> ((dim % 4) * 2)[None, :]) & 3) | (
                    ((bit.to(tl.int32) >> (dim % 8)[None, :]) & 1) << 2
                )
            else:
                byte = tl.load(base + (g * 64 + dim // 2)[None, :])
                idx = (byte.to(tl.int32) >> ((dim % 2) * 4)[None, :]) & 0x0F
            norm = tl.load(norm_ptr + slots * stride_n_token + head * stride_n_head + g)
            rotated = tl.load(cent_ptr + idx) * norm.to(tl.float32)[:, None]
            # Undo the per-group randomized Hadamard: one 128x128 product on the tensor cores.
            value = tl.dot(rotated.to(tl.bfloat16), rot)
            tl.store(
                out_ptr
                + dst * stride_o_page
                + tok[:, None] * stride_o_token
                + head * stride_o_head
                + (g * 128 + dim)[None, :],
                value.to(out_ptr.dtype.element_ty),
            )


def turbo_inverse_rotation(device: torch.device) -> torch.Tensor:
    """``M[i, j] = s2[i] * H[i, j] * s1[j] / sqrt(128)``: ``y @ M`` is ``turbo_kv.inv_rotate(y)``."""
    from freetoken.kernel.triton.turbo_kv import FWHT_SCALE, _signs, hadamard

    s1, s2 = _signs(device)
    had = hadamard(device=device)
    return (s2[:, None] * had * FWHT_SCALE * s1[None, :]).to(torch.bfloat16).contiguous()


def turbo_pages_to_bf16(
    codes: torch.Tensor,
    norm: torch.Tensor,
    cent: torch.Tensor,
    rotation: torch.Tensor,
    src_pages: torch.Tensor,
    dst_pages: torch.Tensor,
    out: torch.Tensor,
    book: str,
    page_size: int,
) -> None:
    """Decode turbo pages ``src_pages`` (``-1`` = skip) of a token-major code/norm slab into the
    original-domain page slab ``out[dst_pages]`` (``[pages, page_size, heads, head_dim]``)."""
    heads, groups = norm.shape[1], norm.shape[2]
    if src_pages.shape != dst_pages.shape or out.shape[1:] != (page_size, heads, groups * 128):
        raise ValueError("turbo page decode needs matching page lists and a page-shaped output")
    if not src_pages.shape[0]:
        return
    _turbo_pages_kernel[(src_pages.shape[0], heads)](
        codes, norm, cent, rotation, src_pages, dst_pages, out,
        codes.stride(0), codes.stride(1), norm.stride(0), norm.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        PAGE_SIZE=page_size, GROUPS=groups, BOOK3=book == "turbo3", num_warps=4,
    )  # fmt: skip


__all__ = [
    "gather_pages",
    "scatter_rows",
    "swap_pages",
    "tiered_store_kv",
    "turbo_inverse_rotation",
    "turbo_pages_to_bf16",
    "zero_tier",
]
