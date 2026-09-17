"""Paged KV pool that stores the full-attention slab as turbo3 / turbo4 codes.

The point is bytes: a head_dim-128 row costs 50 B (turbo3, 3.125 bpv) or 66 B (turbo4, 4.125 bpv)
per KV head per slab instead of 256 B in bf16, so the context the planner has to buy out of the
expert cache shrinks ~3-4x. The read side is the other half of the bargain: values are stored in
the ROTATED domain, so the attention kernel pre-rotates Q and consumes ``centroid * norm`` per tile
without ever de-rotating a tile or materializing a context-sized buffer (see
``kernel/triton/turbo_kv.py`` and its ``test_rotated_domain_scores_match_the_materialized_path``).

``k_cache`` / ``v_cache`` refuse rather than hand back a uint8 view: a backend that expects a bf16
slab must fail here, not compute garbage.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .base import BaseKVCachePool

_ROTATED = "stored rotated-domain; the fused read path consumes k_slab()/v_slab()"


def packed_bytes_per_token(head_dim: int, num_kv_heads: int, book: str, slabs: int = 2) -> int:
    from freetoken.kernel.triton.turbo_kv import CODE_BYTES, QK_TURBO

    if head_dim % QK_TURBO:
        raise ValueError(f"turbo KV needs head_dim a multiple of {QK_TURBO}, got {head_dim}")
    groups = head_dim // QK_TURBO
    return slabs * num_kv_heads * groups * (CODE_BYTES[book] + 2)


class TurboMHAKVCache(BaseKVCachePool):
    """KV storage for the non-SWA groups of a model, quantized with the turbo books.

    Constructor signature mirrors MHAKVCache so the factory can swap between them; ``book``
    selects turbo3 or turbo4. Buffers are per storage layer (the hybrid-linear remap that
    MHAKVCache does with ``layer_ids`` is reused verbatim), token-major, one row group per
    128 head dimensions.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_ids: Sequence[int] | None = None,
        book: str = "turbo4",
    ) -> None:
        from freetoken.kernel.triton.turbo_kv import CODE_BYTES, QK_TURBO, BOOKS

        if book not in BOOKS:
            raise ValueError(f"unknown turbo book {book!r} (known: {', '.join(BOOKS)})")
        if head_dim % QK_TURBO:
            raise ValueError(
                f"turbo KV prices whole 128-element rotation groups; head_dim={head_dim} is not a "
                "multiple of 128. Zero-padding it up would cost more than bf16 KV."
            )
        if head_dim > 512:
            raise ValueError(f"turbo KV supports head_dim <= 512, got {head_dim}")
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"turbo KV quantizes {dtype} activations; expected bf16/fp16")
        self.book = book
        self._dtype = dtype
        self._device = device
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._page_size = page_size
        self._groups = head_dim // QK_TURBO
        self._code_bytes = CODE_BYTES[book]
        self._layer_map: list[int] | None = None
        num_storage_layers = num_layers
        if layer_ids is not None:
            num_storage_layers = len(layer_ids)
            layer_map = [-1] * num_layers
            for dense, global_id in enumerate(layer_ids):
                if global_id < 0 or global_id >= num_layers:
                    raise ValueError(f"KV layer id {global_id} outside [0, {num_layers})")
                layer_map[global_id] = dense
            self._layer_map = layer_map
        self._num_storage_layers = num_storage_layers
        self._alloc(num_pages)

    def _alloc(self, num_pages: int) -> None:
        tokens = num_pages * self._page_size
        self._tokens = tokens
        shape = (self._num_storage_layers, tokens, self._num_kv_heads, self._groups * self._code_bytes)
        nshape = (self._num_storage_layers, tokens, self._num_kv_heads, self._groups)
        self._k_codes = torch.zeros(shape, device=self._device, dtype=torch.uint8)
        self._v_codes = torch.zeros(shape, device=self._device, dtype=torch.uint8)
        self._k_norm = torch.zeros(nshape, device=self._device, dtype=torch.float16)
        self._v_norm = torch.zeros(nshape, device=self._device, dtype=torch.float16)

    # ---- interface the backends use ----------------------------------------------

    def _dense(self, layer_id: int) -> int:
        if self._layer_map is None:
            return layer_id
        dense = self._layer_map[layer_id]
        if dense < 0:
            raise KeyError(f"layer {layer_id} has no paged KV storage")
        return dense

    @property
    def compressed(self) -> bool:
        return True

    def k_cache(self, index: int) -> torch.Tensor:
        raise NotImplementedError(f"turbo KV holds no bf16 K slab ({_ROTATED})")

    def v_cache(self, index: int) -> torch.Tensor:
        raise NotImplementedError(f"turbo KV holds no bf16 V slab ({_ROTATED})")

    def k_slab(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        dense = self._dense(index)
        return self._k_codes[dense], self._k_norm[dense]

    def v_slab(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        dense = self._dense(index)
        return self._v_codes[dense], self._v_norm[dense]

    def store_kv(self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int) -> None:
        from freetoken.kernel.triton.turbo_kv import quantize

        dense = self._dense(layer_id)
        rows = out_loc.numel()
        stride = self._code_bytes * self._groups
        k2 = k.reshape(rows, self._num_kv_heads, self._head_dim)
        v2 = v.reshape(rows, self._num_kv_heads, self._head_dim)
        kq, kn = quantize(k2.reshape(-1, self._head_dim).to(self._dtype), self.book)
        vq, vn = quantize(v2.reshape(-1, self._head_dim).to(self._dtype), self.book)
        self._k_codes[dense].index_copy_(0, out_loc, kq.reshape(rows, self._num_kv_heads, stride))
        self._k_norm[dense].index_copy_(0, out_loc, kn.reshape(rows, self._num_kv_heads, self._groups))
        self._v_codes[dense].index_copy_(0, out_loc, vq.reshape(rows, self._num_kv_heads, stride))
        self._v_norm[dense].index_copy_(0, out_loc, vn.reshape(rows, self._num_kv_heads, self._groups))

    def decode_rows(self, layer_id: int, which: str = "k") -> torch.Tensor:
        """Materialize the whole slab. Only for tests and the measurement arm that quantifies how
        much the fused path saves -- never on a serving hot path."""
        from freetoken.kernel.triton.turbo_kv import decode_rotated, inv_rotate

        codes, norm = (self._k_codes, self._k_norm) if which == "k" else (self._v_codes, self._v_norm)
        dense = self._dense(layer_id)
        tokens, heads = codes.shape[1], codes.shape[2]
        flat = codes[dense].reshape(-1, self._code_bytes)
        nflat = norm[dense].reshape(-1, 1)
        y = decode_rotated(flat, nflat, self.book)
        return inv_rotate(y).reshape(tokens, heads, self._head_dim).to(self._dtype)

    # ---- sizing ------------------------------------------------------------------

    @classmethod
    def kv_cost(cls, config, **kwargs) -> tuple[int, int, int, int]:
        book = getattr(config, "kv_book", None) or kwargs.get("book") or "turbo4"
        per_token = 0
        for spec in config.model_config.kv_cache_group_specs():
            if spec.is_swa:
                continue
            heads = _local_heads(spec, config)
            per_token += packed_bytes_per_token(spec.head_dim, heads, book) * spec.num_layers
        return per_token * config.page_size, 0, config.page_size, 0

    def rebuild_from_config(self, config, num_pages: int, *, num_swa_pages: int | None = None) -> None:
        self._alloc(num_pages + 1)  # +1 for the dummy page (matches create_kvcache_pool)

    def unit_bytes(self) -> tuple[int, int]:
        per_token = (
            self._k_codes.numel()
            + self._v_codes.numel()
            + self._k_norm.numel() * self._k_norm.element_size()
            + self._v_norm.numel() * self._v_norm.element_size()
        ) // self._tokens
        return per_token, 0

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def total_bytes(self) -> int:
        return (
            self._k_codes.numel()
            + self._v_codes.numel()
            + self._k_norm.numel() * 2
            + self._v_norm.numel() * 2
        )


def _local_heads(spec, config) -> int:
    from freetoken.utils import div_even

    tp = getattr(getattr(config, "tp_info", None), "size", None)
    if tp is None:
        from freetoken.distributed import get_tp_info

        tp = get_tp_info().size
    return div_even(spec.num_kv_heads, tp, allow_replicate=True)
