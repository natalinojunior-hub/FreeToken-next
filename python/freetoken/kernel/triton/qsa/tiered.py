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
        tl.store(k_host_ptr + row * ROW + cols, k, mask=mask & on_host)
        tl.store(v_host_ptr + row * ROW + cols, v, mask=mask & on_host)


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
        if not t.is_contiguous() or t.dtype != k.dtype:
            raise ValueError("tiered KV slabs must be contiguous and match the K/V dtype")
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
        tl.load(src_ptr + page * PAGE_ELEMS + cols, mask=mask),
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


__all__ = ["gather_pages", "tiered_store_kv", "zero_tier"]
