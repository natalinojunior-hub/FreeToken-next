# SPDX-License-Identifier: Apache-2.0
"""Triton kernel to decompress Turbo3 / Turbo4 KV cache into a dense BF16/FP16 workspace."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _decompress_turbo_paged_kernel(
    k_codes_ptr,
    k_norm_ptr,
    v_codes_ptr,
    v_norm_ptr,
    cent_ptr,
    block_table_ptr,
    seq_lens_ptr,
    workspace_k_ptr,
    workspace_v_ptr,
    stride_kc_token,
    stride_kc_head,
    stride_kn_token,
    stride_kn_head,
    stride_vc_token,
    stride_vc_head,
    stride_vn_token,
    stride_vn_head,
    stride_ws_block,
    stride_ws_token,
    stride_ws_head,
    stride_table_req,
    num_cache_blocks,
    selected_pages_ptr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUPS: tl.constexpr,
    BOOK3: tl.constexpr = False,
    USE_SELECTED: tl.constexpr = False,
):
    block_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    req_idx = tl.program_id(2)

    if USE_SELECTED:
        page_id = tl.load(selected_pages_ptr + block_idx)
    else:
        seq_len = tl.load(seq_lens_ptr + req_idx)
        if block_idx * PAGE_SIZE >= seq_len:
            return
        page_id = tl.load(block_table_ptr + req_idx * stride_table_req + block_idx)

    if page_id < 0 or page_id >= num_cache_blocks:
        return

    token_offs = tl.arange(0, PAGE_SIZE)
    dim_offs = tl.arange(0, 128)
    slots = page_id.to(tl.int64) * PAGE_SIZE + token_offs

    if BOOK3:
        for g in range(GROUPS):
            low_k = tl.load(
                k_codes_ptr
                + slots[:, None] * stride_kc_token
                + head_idx * stride_kc_head
                + (g * 48 + dim_offs // 4)[None, :]
            )
            bit_k = tl.load(
                k_codes_ptr
                + slots[:, None] * stride_kc_token
                + head_idx * stride_kc_head
                + (g * 48 + 32 + dim_offs // 8)[None, :]
            )
            k_idx = ((low_k.to(tl.int32) >> ((dim_offs % 4) * 2)[None, :]) & 3) | (
                ((bit_k.to(tl.int32) >> (dim_offs % 8)[None, :]) & 1) << 2
            )
            kn = tl.load(
                k_norm_ptr + slots * stride_kn_token + head_idx * stride_kn_head + g
            ).to(tl.float32)
            k_val = (tl.load(cent_ptr + k_idx) * kn[:, None]).to(tl.bfloat16)
            tl.store(
                workspace_k_ptr
                + page_id.to(tl.int64) * stride_ws_block
                + token_offs[:, None] * stride_ws_token
                + head_idx * stride_ws_head
                + (g * 128 + dim_offs)[None, :],
                k_val,
            )

            low_v = tl.load(
                v_codes_ptr
                + slots[:, None] * stride_vc_token
                + head_idx * stride_vc_head
                + (g * 48 + dim_offs // 4)[None, :]
            )
            bit_v = tl.load(
                v_codes_ptr
                + slots[:, None] * stride_vc_token
                + head_idx * stride_vc_head
                + (g * 48 + 32 + dim_offs // 8)[None, :]
            )
            v_idx = ((low_v.to(tl.int32) >> ((dim_offs % 4) * 2)[None, :]) & 3) | (
                ((bit_v.to(tl.int32) >> (dim_offs % 8)[None, :]) & 1) << 2
            )
            vn = tl.load(
                v_norm_ptr + slots * stride_vn_token + head_idx * stride_vn_head + g
            ).to(tl.float32)
            v_val = (tl.load(cent_ptr + v_idx) * vn[:, None]).to(tl.bfloat16)
            tl.store(
                workspace_v_ptr
                + page_id.to(tl.int64) * stride_ws_block
                + token_offs[:, None] * stride_ws_token
                + head_idx * stride_ws_head
                + (g * 128 + dim_offs)[None, :],
                v_val,
            )
    else:
        byte_offs = dim_offs // 2
        shift = (dim_offs % 2) * 4
        for g in range(GROUPS):
            c_offs = g * 64 + byte_offs

            kc = tl.load(
                k_codes_ptr
                + slots[:, None] * stride_kc_token
                + head_idx * stride_kc_head
                + c_offs[None, :]
            )
            kn = tl.load(
                k_norm_ptr
                + slots * stride_kn_token
                + head_idx * stride_kn_head
                + g
            ).to(tl.float32)

            k_idx = (kc.to(tl.int32) >> shift[None, :]) & 0x0F
            k_val = (tl.load(cent_ptr + k_idx) * kn[:, None]).to(tl.bfloat16)

            tl.store(
                workspace_k_ptr
                + page_id.to(tl.int64) * stride_ws_block
                + token_offs[:, None] * stride_ws_token
                + head_idx * stride_ws_head
                + (g * 128 + dim_offs)[None, :],
                k_val,
            )

            vc = tl.load(
                v_codes_ptr
                + slots[:, None] * stride_vc_token
                + head_idx * stride_vc_head
                + c_offs[None, :]
            )
            vn = tl.load(
                v_norm_ptr
                + slots * stride_vn_token
                + head_idx * stride_vn_head
                + g
            ).to(tl.float32)

            v_idx = (vc.to(tl.int32) >> shift[None, :]) & 0x0F
            v_val = (tl.load(cent_ptr + v_idx) * vn[:, None]).to(tl.bfloat16)

            tl.store(
                workspace_v_ptr
                + page_id.to(tl.int64) * stride_ws_block
                + token_offs[:, None] * stride_ws_token
                + head_idx * stride_ws_head
                + (g * 128 + dim_offs)[None, :],
                v_val,
            )


def decompress_turbo4_to_workspace(
    k_codes: torch.Tensor,
    k_norm: torch.Tensor,
    v_codes: torch.Tensor,
    v_norm: torch.Tensor,
    cent: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_k: torch.Tensor,
    workspace_v: torch.Tensor,
    book3: bool = False,
    selected_pages: torch.Tensor | None = None,
) -> None:
    """Decompress Turbo3 / Turbo4 KV pages into dense workspace buffers.
    
    When `selected_pages` is provided (1-D int32 tensor of physical page ids), only
    those active pages are decompressed, bounding decode decompression cost to O(1)
    with respect to context length.
    """
    num_requests = block_table.shape[0]
    max_blocks = block_table.shape[1]
    if num_requests == 0:
        return

    num_kv_heads = k_codes.shape[1]
    page_size = workspace_k.shape[1]
    head_dim = workspace_k.shape[3]
    groups = head_dim // 128
    num_cache_blocks = workspace_k.shape[0]

    use_selected = selected_pages is not None
    if use_selected:
        num_blocks = selected_pages.numel()
        if num_blocks == 0:
            return
        grid = (num_blocks, num_kv_heads, 1)
        sel_ptr = selected_pages
    else:
        if max_blocks == 0:
            return
        grid = (max_blocks, num_kv_heads, num_requests)
        sel_ptr = block_table  # unused when USE_SELECTED is False

    _decompress_turbo_paged_kernel[grid](
        k_codes,
        k_norm,
        v_codes,
        v_norm,
        cent,
        block_table,
        seq_lens,
        workspace_k,
        workspace_v,
        k_codes.stride(0),
        k_codes.stride(1),
        k_norm.stride(0),
        k_norm.stride(1),
        v_codes.stride(0),
        v_codes.stride(1),
        v_norm.stride(0),
        v_norm.stride(1),
        workspace_k.stride(0),
        workspace_k.stride(1),
        workspace_k.stride(2),
        block_table.stride(0),
        num_cache_blocks,
        sel_ptr,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        NUM_KV_HEADS=num_kv_heads,
        GROUPS=groups,
        BOOK3=book3,
        USE_SELECTED=use_selected,
        num_warps=4,
        num_stages=1,
    )


__all__ = ["decompress_turbo4_to_workspace"]
