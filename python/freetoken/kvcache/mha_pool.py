from __future__ import annotations

import weakref
from typing import Sequence

import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import div_even

from .base import BaseKVCachePool


def registered_host_empty(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Page-locked host tensor the GPU reads zero-copy through its host address (UVA).

    ``pin_memory=True`` rounds to power-of-two blocks (a 5 GiB tier would pin 8 GiB), so the
    storage is a plain allocation registered for the tensor's lifetime instead."""
    host = torch.empty(shape, dtype=dtype)
    cudart = torch.cuda.cudart()
    err = cudart.cudaHostRegister(host.data_ptr(), host.numel() * host.element_size(), 2)
    if err != cudart.cudaError.success:
        raise MemoryError(f"cannot page-lock {host.nbytes} bytes for the KV RAM tier: {err}")
    weakref.finalize(host, cudart.cudaHostUnregister, host.data_ptr())
    return host


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.

    ``layer_ids`` lets the pool back only a *subset* of the model's layers while
    callers keep indexing by their global ``layer_id``. Hybrid models (e.g. the
    Qwen3.5 GatedDeltaNet/full-attention stack) interleave linear-attention layers
    that hold no paged KV; passing the full-attention layer ids here allocates one
    storage slab per KV layer (not per model layer) and remaps the global id to its
    dense slot, avoiding a multiple-x over-allocation of unused slabs.
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
        host_pages: int = 0,
        host_dtype: torch.dtype | str | None = None,
    ) -> None:
        """``host_pages`` of the ``num_pages`` physical pages (the highest ids) live in the
        page-locked RAM tier (in ``host_dtype``, default the KV dtype; FP8 halves it); the
        rest form the device slab."""
        if not 0 <= host_pages < num_pages:
            raise ValueError(f"host_pages ({host_pages}) must leave device pages of {num_pages}")
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._num_layers = num_layers
        if layer_ids is None:
            num_storage_layers = num_layers
            self._layer_map: list[int] | None = None
        else:
            num_storage_layers = len(layer_ids)
            layer_map = [-1] * num_layers
            for dense, global_id in enumerate(layer_ids):
                if global_id < 0 or global_id >= num_layers:
                    raise ValueError(f"KV layer id {global_id} outside [0, {num_layers})")
                layer_map[global_id] = dense
            self._layer_map = layer_map
        self._host_pages = host_pages
        num_pages -= host_pages
        self._kv_buffer = torch.empty(
            (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        # RAM tier: element-wise (BF16/FP8, read in place) or turbo codes + norms (decoded page
        # by page into device staging before attention reads them).
        self.host_book = host_dtype if host_dtype in ("turbo4", "turbo3") else None
        self._kv_host = self._host_codes = self._host_norm = None
        if host_pages and self.host_book is not None:
            from freetoken.kernel.triton.turbo_kv import CODE_BYTES

            groups = head_dim // 128
            rows = (2, num_storage_layers, host_pages * page_size, local_kv_heads)
            self._host_codes = registered_host_empty(
                (*rows, groups * CODE_BYTES[self.host_book]), torch.uint8
            )
            self._host_norm = registered_host_empty((*rows, groups), torch.float16)
        elif host_pages:
            self._kv_host = registered_host_empty(
                (2, num_storage_layers, host_pages, page_size, local_kv_heads, head_dim),
                host_dtype if isinstance(host_dtype, torch.dtype) else dtype,
            )
        self._host_pages_count = host_pages
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    @classmethod
    def host_tier_device_bytes(cls, config, host_tokens: int) -> int:
        # MHA host pages are read directly; only optional staging is charged.
        return 0

    @classmethod
    def host_tier_ram_bytes(cls, config, host_tokens: int, dtype=None) -> int:
        from freetoken.attention import AttnType
        from freetoken.utils import div_even
        total = 0
        for spec in config.model_config.kv_cache_group_specs():
            if spec.attn_type is not AttnType.FULL or getattr(spec, "is_swa", False):
                continue
            heads = div_even(spec.num_kv_heads, config.tp_info.size, allow_replicate=True)
            if dtype in ("turbo4", "turbo3"):
                from freetoken.kernel.triton.turbo_kv import CODE_BYTES
                per_head = (spec.head_dim // 128) * (CODE_BYTES[dtype] + 2)
            else:
                itemsize = dtype.itemsize if isinstance(dtype, torch.dtype) else config.dtype.itemsize
                per_head = spec.head_dim * itemsize
            total += 2 * spec.num_layers * heads * per_head
        return total * host_tokens

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the KV buffer for ``num_pages`` pages IN PLACE.

        Geometry (storage layers, page_size, kv heads, head_dim) is taken from the
        existing buffer; only the page count changes. Views and ``_storage_shape`` are
        refreshed. Object identity is preserved so cached backend references stay valid.
        ``num_pages`` counts the RAM tier too; the tier itself keeps its size.
        """
        if num_pages <= self._host_pages:
            raise ValueError(f"num_pages ({num_pages}) must exceed the RAM tier")
        num_pages -= self._host_pages
        _, num_storage_layers, _old_pages, page_size, local_kv_heads, head_dim = (
            self._kv_buffer.shape
        )
        dtype = self._kv_buffer.dtype
        device = self._device
        self._k_buffer = None
        self._v_buffer = None
        self._kv_buffer = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self._kv_buffer = torch.empty(
            (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token

        per_token = sum(
            spec_kv_bytes_per_token(spec, config)
            for spec in config.model_config.kv_cache_group_specs()
            if not spec.is_swa
        )
        return per_token * config.page_size, 0, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        self.rebuild(num_pages + 1)  # +1 for the dummy page (matches create_kvcache_pool)

    def unit_bytes(self) -> tuple[int, int]:
        buf = self._kv_buffer
        tokens = int(buf.shape[2]) * int(buf.shape[3])
        return int(buf.numel() * buf.element_size()) // tokens, 0

    def _dense(self, layer_id: int) -> int:
        if self._layer_map is None:
            return layer_id
        dense = self._layer_map[layer_id]
        if dense < 0:
            raise KeyError(f"layer {layer_id} has no paged KV storage")
        return dense

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[self._dense(index)]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[self._dense(index)]

    def host_turbo(self, index: int) -> tuple[torch.Tensor, ...] | None:
        """One layer's turbo RAM tier: ``(k_codes, k_norm, v_codes, v_norm)``, token-major."""
        if self.host_book is None:
            return None
        dense = self._dense(index)
        return (
            self._host_codes[0, dense],
            self._host_norm[0, dense],
            self._host_codes[1, dense],
            self._host_norm[1, dense],
        )

    def _store_turbo_tier(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, dense: int
    ) -> None:
        """Device slots take BF16 rows; RAM slots take turbo codes of the same rows (quantized
        for every row, fixed shapes, so the path stays graph-capturable)."""
        from freetoken.kernel.triton.qsa.tiered import scatter_rows, tiered_store_kv
        from freetoken.kernel.triton.turbo_kv import quantize

        k_dev, v_dev = self._k_buffer[dense], self._v_buffer[dense]
        tiered_store_kv(k, v, out_loc, (k_dev, v_dev), (k_dev[:0], v_dev[:0]))
        rows = out_loc.shape[0]
        heads, dim = k_dev.shape[2], k_dev.shape[3]
        base = self.num_device_pages * k_dev.shape[1]
        for side, x in enumerate((k, v)):
            codes, norm = quantize(x.reshape(rows * heads, dim), self.host_book)
            scatter_rows(
                codes.reshape(rows, heads, -1), out_loc, self._host_codes[side, dense], base
            )
            scatter_rows(norm.reshape(rows, heads, -1), out_loc, self._host_norm[side, dense], base)

    @property
    def num_device_pages(self) -> int:
        return int(self._kv_buffer.shape[2])

    def host_kv(self, index: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """One layer's RAM-tier K/V pages, addressed as ``page - num_device_pages``."""
        if self._kv_host is None:
            return None
        dense = self._dense(index)
        return self._kv_host[0, dense], self._kv_host[1, dense]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        from freetoken.kernel import store_cache

        dense = self._dense(layer_id)
        if self.host_book is not None:
            self._store_turbo_tier(k, v, out_loc, dense)
            return
        if self._kv_host is not None:
            from freetoken.kernel.triton.qsa.tiered import tiered_store_kv

            tiered_store_kv(
                k,
                v,
                out_loc,
                (self._k_buffer[dense], self._v_buffer[dense]),
                (self._kv_host[0, dense], self._kv_host[1, dense]),
            )
            return
        store_cache(
            k_cache=self._k_buffer[dense].view(self._storage_shape),
            v_cache=self._v_buffer[dense].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    def attach_staging(self, binding: object) -> None:
        """Attach stable logical-page indirection for opt-in tiered launches."""
        if getattr(binding, "device", self._device) != self._device:
            raise ValueError("staging binding must use the KV pool device")
        self._kv_staging = binding

    def staging_table(self) -> torch.Tensor | None:
        """Return the graph-stable logical-to-physical table, when attached."""
        binding = getattr(self, "_kv_staging", None)
        return None if binding is None else binding.table

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
