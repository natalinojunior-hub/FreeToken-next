# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM (vllm/models/qwen4_exp/nvidia/ops/qsa.py)
"""Sparse paged GQA over the QSA selection."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from freetoken.kernel.triton.turbo_attn import (
    BOOK_CODE,
    turbo_k_tile,
    turbo_v_tile,
)


@triton.jit
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_norm_ptr,
    v_norm_ptr,
    cent_ptr,
    k_host_ptr,
    v_host_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_kn_token,
    stride_kn_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_vn_token,
    stride_vn_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_cache_blocks,
    num_host_blocks,
    num_requests,
    TIERED: tl.constexpr,
    HOST_CAST: tl.constexpr,
    CODED: tl.constexpr,
    BOOK: tl.constexpr,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    # row * stride can overflow int32 for large row counts.
    row = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    # Dynamic bounds avoid padded main-loop iterations for uneven splits.
    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request.to(tl.int64) * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        if TIERED:
            # Pages past the device slab live in the host tier (pinned zero-copy memory, or a
            # device staging copy of it); both tiers share one page layout and stride set.
            valid &= (physical_page >= 0) & (physical_page < num_cache_blocks + num_host_blocks)
            on_device = physical_page < num_cache_blocks
            safe_page = tl.where(on_device, physical_page, physical_page - num_cache_blocks)
            safe_page = tl.maximum(safe_page, 0).to(tl.int64)
            key_offsets = (
                safe_page[None, :] * stride_k_block
                + page_offset[None, :] * stride_k_token
                + kv_head * stride_k_head
                + dim_offsets[:, None]
            )
            value_offsets = (
                safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
                + dim_offsets[None, :]
            )
            if CODED:
                slots = safe_page * PAGE_SIZE + page_offset
                device_keys = turbo_k_tile(
                    k_cache_ptr,
                    k_norm_ptr,
                    cent_ptr,
                    slots,
                    kv_head,
                    stride_k_token,
                    stride_k_head,
                    stride_kn_token,
                    stride_kn_head,
                    dim_offsets,
                    valid & on_device,
                    BOOK,
                    query.dtype,
                )
                device_values = turbo_v_tile(
                    v_cache_ptr,
                    v_norm_ptr,
                    cent_ptr,
                    slots,
                    kv_head,
                    stride_v_token,
                    stride_v_head,
                    stride_vn_token,
                    stride_vn_head,
                    dim_offsets,
                    valid & on_device,
                    BOOK,
                    query.dtype,
                )
                host_keys = tl.load(
                    k_host_ptr + key_offsets,
                    mask=(valid & ~on_device)[None, :],
                    other=0.0,
                ).to(query.dtype)
                host_values = tl.load(
                    v_host_ptr + value_offsets,
                    mask=(valid & ~on_device)[:, None],
                    other=0.0,
                ).to(query.dtype)
                keys = device_keys + host_keys
                values = device_values + host_values
            elif HOST_CAST:
                # Narrower RAM tier (FP8): two masked loads, the host one widened.
                keys = tl.load(
                    k_cache_ptr + key_offsets, mask=(valid & on_device)[None, :], other=0.0
                )
                keys += tl.load(
                    k_host_ptr + key_offsets, mask=(valid & ~on_device)[None, :], other=0.0
                ).to(keys.dtype)
                values = tl.load(
                    v_cache_ptr + value_offsets, mask=(valid & on_device)[:, None], other=0.0
                )
                values += tl.load(
                    v_host_ptr + value_offsets, mask=(valid & ~on_device)[:, None], other=0.0
                ).to(values.dtype)
            else:
                # One load through a per-lane base pointer: two masked loads would double the
                # pipelined shared-memory footprint and overflow it on 100 KB/SM parts.
                keys = tl.load(
                    tl.where(on_device[None, :], k_cache_ptr, k_host_ptr) + key_offsets,
                    mask=valid[None, :],
                    other=0.0,
                )
                values = tl.load(
                    tl.where(on_device[:, None], v_cache_ptr, v_host_ptr) + value_offsets,
                    mask=valid[:, None],
                    other=0.0,
                )
        else:
            valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
            # physical_page * block stride can overflow int32 for large caches.
            safe_page = tl.maximum(physical_page, 0).to(tl.int64)
            if CODED:
                slots = safe_page * PAGE_SIZE + page_offset
                keys = turbo_k_tile(
                    k_cache_ptr,
                    k_norm_ptr,
                    cent_ptr,
                    slots,
                    kv_head,
                    stride_k_token,
                    stride_k_head,
                    stride_kn_token,
                    stride_kn_head,
                    dim_offsets,
                    valid,
                    BOOK,
                    query.dtype,
                )
                values = turbo_v_tile(
                    v_cache_ptr,
                    v_norm_ptr,
                    cent_ptr,
                    slots,
                    kv_head,
                    stride_v_token,
                    stride_v_head,
                    stride_vn_token,
                    stride_vn_head,
                    dim_offsets,
                    valid,
                    BOOK,
                    query.dtype,
                )
            else:
                keys = tl.load(
                    k_cache_ptr
                    + safe_page[None, :] * stride_k_block
                    + page_offset[None, :] * stride_k_token
                    + kv_head * stride_k_head
                    + dim_offsets[:, None],
                    mask=valid[None, :],
                    other=0.0,
                )
                values = tl.load(
                    v_cache_ptr
                    + safe_page[:, None] * stride_v_block
                    + page_offset[:, None] * stride_v_token
                    + kv_head * stride_v_head
                    + dim_offsets[None, :],
                    mask=valid[:, None],
                    other=0.0,
                )
        scores = tl.dot(query, keys)
        # Scaling scores avoids re-quantizing a scaled query to BF16.
        scores *= softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0)
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    output_mask = head_offsets[:, None] < GROUP_SIZE
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    if NUM_SPLITS == 1:
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        tl.store(
            partial_output_ptr
            + (
                (split_id.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS
                + first_head
                + head_offsets[:, None]
            )
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr
            + (split_id.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS
            + first_head
            + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_output_row,
    stride_output_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    partial_output = tl.load(
        partial_output_ptr
        + ((split_offsets[:, None].to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS + head)
        * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
    host_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    kv_book: str | None = None,
    kv_norms: tuple[torch.Tensor, torch.Tensor] | None = None,
    cent: torch.Tensor | None = None,
    page_size: int | None = None,
    row_invariant: bool = False,
) -> torch.Tensor:
    """Run sparse GQA directly over paged BF16 K/V caches.

    ``host_kv`` is the RAM tier: physical pages ``>= k_cache.shape[0]`` address it at
    ``page - k_cache.shape[0]``. It is pinned host memory read zero-copy, or a device staging
    copy of it; either way it must share the device slab's page layout.

    ``row_invariant`` picks the tile/split profile as for a single decode row, so every row of a
    multi-row spec-verify window reduces in the same order as RAW decode would.
    """

    coded = kv_book is not None
    if q.ndim != 3 or k_cache.ndim != (3 if coded else 4) or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if logical_indices.shape[1] <= 0:
        raise ValueError("QSA sparse attention requires a positive selection width")
    kv_heads = k_cache.shape[1] if coded else k_cache.shape[2]
    if (not coded and q.shape[2] != k_cache.shape[3]) or q.shape[1] % kv_heads:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    if coded:
        if kv_book not in BOOK_CODE or kv_norms is None or cent is None or page_size is None:
            raise ValueError(
                "coded QSA attention needs a supported format, norms, codebook, and page size"
            )
        assert k_cache.dtype == v_cache.dtype == torch.uint8
        assert kv_norms[0].shape[:2] == kv_norms[1].shape[:2] == k_cache.shape[:2]
    else:
        assert q.dtype == k_cache.dtype == v_cache.dtype
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    if not coded:
        assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1
    if out is None:
        out = torch.empty_like(q)
    assert out.shape == q.shape and out.dtype == q.dtype and out.stride(2) == 1
    if host_kv is not None and coded and kv_book != "fp8":
        raise ValueError("RAM tiering for coded QSA currently requires fp8")
    if host_kv is None:
        k_host, v_host, num_host_blocks = k_cache, v_cache, 0
    else:
        k_host, v_host = host_kv
        if coded:
            if kv_book != "fp8" or k_host.ndim != 4 or k_host.shape[1] != int(page_size):
                raise ValueError("coded QSA RAM tier requires page-major FP8 host KV")
            if k_host.shape[2:] != (kv_heads, head_dim) or v_host.shape != k_host.shape:
                raise ValueError("QSA FP8 host tier must share the device head geometry")
        elif k_host.shape[1:] != k_cache.shape[1:] or v_host.shape != k_host.shape:
            raise ValueError("QSA host KV tier must share the device page layout")
        if not coded and (
            k_host.stride() != k_cache.stride() or v_host.stride() != v_cache.stride()
        ):
            raise ValueError("QSA host KV tier must share the device page strides")
        assert k_host.dtype == v_host.dtype
        assert k_host.dtype in (
            (torch.float8_e4m3fn,) if coded else (k_cache.dtype, torch.float8_e4m3fn)
        )
        num_host_blocks = k_host.shape[0]
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // kv_heads
    block_m = triton.next_power_of_2(group_size)
    base_programs = (1 if row_invariant else q.shape[0]) * kv_heads
    small_profile_limit = 8 if block_m <= 8 else 4

    # Tuned on GB300 for the Qwen-Air TP1, TP2, and TP4 attention shapes.
    # Narrow tiles favor decode; wide tiles improve throughput for prefill.
    if base_programs <= small_profile_limit:
        block_n, target_splits, partial_warps = 16, 64, 4
    elif base_programs < 32:
        block_n, target_splits, partial_warps = 16, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, partial_warps = 64, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, partial_warps = 64, 4, 2
    else:
        block_n, target_splits, partial_warps = 64, 1, 2

    if coded:
        # Decoding packed bytes in the attention tile increases live registers/shared state.
        block_n, partial_warps = 16, 4

    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)
    # Avoid empty splits when the selection width is smaller than the profile.
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    num_splits = min(max_useful_splits, target_splits)

    # Split=1 writes output directly and compiles out all workspace accesses.
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        # FP32 partials preserve accuracy when merging independently normalized
        # splits.
        partial_output = torch.empty((num_splits, *q.shape), dtype=torch.float32, device=q.device)
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    partial_grid = (q.shape[0], kv_heads, num_splits)
    if coded:
        k_norm, v_norm = kv_norms
        stride_k_block = int(page_size) * k_cache.stride(0)
        stride_v_block = int(page_size) * v_cache.stride(0)
        num_cache_blocks = k_cache.shape[0] // int(page_size)
    else:
        k_norm = v_norm = k_cache
        stride_k_block = k_cache.stride(0)
        stride_v_block = v_cache.stride(0)
        num_cache_blocks = k_cache.shape[0]
    _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
        q,
        k_cache,
        v_cache,
        k_norm,
        v_norm,
        cent if cent is not None else k_cache,
        k_host,
        v_host,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        q.stride(0),
        q.stride(1),
        stride_k_block,
        k_cache.stride(0) if coded else k_cache.stride(1),
        k_cache.stride(1) if coded else k_cache.stride(2),
        k_norm.stride(0) if coded else 0,
        k_norm.stride(1) if coded else 0,
        stride_v_block,
        v_cache.stride(0) if coded else v_cache.stride(1),
        v_cache.stride(1) if coded else v_cache.stride(2),
        v_norm.stride(0) if coded else 0,
        v_norm.stride(1) if coded else 0,
        logical_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        q.shape[0],
        num_cache_blocks,
        num_host_blocks,
        block_table.shape[0],
        TIERED=host_kv is not None,
        HOST_CAST=k_host.dtype != k_cache.dtype,
        CODED=coded,
        BOOK=BOOK_CODE[kv_book] if coded else 0,
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=int(page_size) if coded else k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=partial_warps,
        # The tiered gather cannot use the async-copy pipeline; two stages of it overflow
        # 100 KB/SM shared memory on the wide prefill tiles.
        num_stages=1 if coded or (host_kv is not None and block_n > 16) else 2,
    )
    if num_splits == 1:
        return out

    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_lse,
        out,
        out.stride(0),
        out.stride(1),
        q.shape[0],
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=2,
        num_stages=1,
    )
    return out


__all__ = ["qsa_sparse_paged_attention"]
