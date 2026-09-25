"""QSA compressed-block sparse KV pool: paged GQA K/V + compressed index keys + pending ring.

Qwen3.8-Flash-Next scores whole ``index_ratio``-token groups instead of single tokens, so
its indexer slab holds ONE compressed key row per group, addressed by ``slot //
index_ratio``. Because ``page_size % index_ratio == 0``, a group's tokens always live in one
page at consecutive slots, which makes that division well-defined: the compressed rows are a
1/ratio shadow of the K/V pages and follow page sharing and eviction for free -- no
allocator, no free, no clear (SGLang qsa_kv_pool / vLLM compressed-region precedent).

Two tiers ride alongside the shadow slab and are NOT per-token:
- ``pending_ring``: the last ``ring_capacity`` pre-RoPE index keys of each running request (sized by ``ring_capacity_for``), indexed by ``Req.table_idx``. A group that straddles two forwards (chunked prefill, and
  every decode step) reads its already-consumed members from here. Never cleared: a new
  tenant of a table_idx starts at a group boundary (cached_len is 0 or a page multiple), so
  its first closing group takes every member from its own forward.
- scratch rows at ``cmp_scratch_base``: one row per request slot, the write target for rows
  whose group does not close in this forward, so the compress kernel scatters unconditionally
  with no negative index and no cross-row conflict (DSV4 precedent).

The slab is amortized into the per-token KV price (``unit_bytes``); the ring and scratch are
fixed and priced through ``kv_cost``'s ``fixed_cache_size``.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .mha_pool import MHAKVCache

# The index tiers are always 2-byte (compute dtype); spec_kv_bytes_per_token budgets the same.
_INDEX_DTYPE_BYTES = 2
# t/h/w int32 rope position kept per KV slot on mrope models
_ROPE_POS_BYTES = 3 * 4


from .base import BaseKVCachePool


class QSAKVCache(BaseKVCachePool):
    """MHA paged pool + the compressed index-key slab + the per-request pending ring.

    ``cmp_k_cache(slot)`` is row-flat ``[num_pages * page_size // index_ratio + num_req_slots,
    index_head_dim]``: row ``r < cmp_scratch_base`` holds the compressed key of the token group
    whose K/V slots are ``[r * index_ratio, (r + 1) * index_ratio)``, and the rows from
    ``cmp_scratch_base`` on are the per-request-slot scratch sinks. ``slot`` is the sparse
    layer's order in the attention backend, same convention as BSAKVCache/DSAKVCache.
    """

    @classmethod
    def ring_capacity_for(cls, index_ratio: int, num_speculative_tokens: int = 0) -> int:
        """Ring depth: one row per pending position, keyed ``position % capacity``; spec decode widens by the draft depth (vLLM sizing)."""
        return index_ratio * math.ceil((index_ratio + num_speculative_tokens) / index_ratio)

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        index_head_dim: int,
        num_index_layers: int,
        index_ratio: int,
        num_req_slots: int,
        ring_capacity: int | None = None,
        layer_ids: Sequence[int] | None = None,
        mrope: bool = False,
        kv_format: str = "auto",
        mtp_layer_id: int | None = None,
        tcq_policy=None,
        host_pages: int = 0,
    ) -> None:
        if index_ratio < 1 or page_size % index_ratio != 0:
            # slot // index_ratio only names one group when a group never straddles a page.
            raise ValueError(
                f"QSA needs page_size ({page_size}) divisible by index_ratio ({index_ratio})"
            )
        if ring_capacity is None:
            ring_capacity = self.ring_capacity_for(index_ratio)
        if ring_capacity < index_ratio:
            # A closing group reads up to index_ratio - 1 past members plus this forward's.
            raise ValueError(
                f"QSA needs ring_capacity ({ring_capacity}) >= index_ratio ({index_ratio})"
            )
        # Index keys ride the compute dtype (the model's index_k is engine-dtype). The KV cost
        # model budgets 2 bytes per token per index layer for the slab
        # (base.spec_kv_bytes_per_token); keep the two in lockstep.
        assert dtype.itemsize == _INDEX_DTYPE_BYTES, (
            f"QSA index slab budgets 2 bytes/token (spec_kv_bytes_per_token); got {dtype}"
        )
        self._index_head_dim = index_head_dim
        self._num_index_layers = num_index_layers
        self._index_ratio = index_ratio
        self._num_req_slots = num_req_slots
        self._ring_capacity = ring_capacity
        self._index_dtype = dtype
        self._page_size = page_size
        self._mrope = mrope
        compressed = kv_format in ["turbo3", "turbo4", "vbr", "tcq"] or tcq_policy is not None
        if host_pages and compressed:
            # The turbo decompress kernel addresses one device code/norm slab; fail closed.
            raise NotImplementedError(
                f"KV RAM tier supports the BF16 QSA layout only, not kv_format={kv_format!r}"
            )
        if compressed:
            from .turbo_pool import TurboMHAKVCache
            from .tcq_policy import TCQPolicy

            if tcq_policy is not None or kv_format in ["vbr", "tcq"]:
                if isinstance(tcq_policy, TCQPolicy):
                    policy = tcq_policy
                else:
                    vbr_pol = tcq_policy if isinstance(tcq_policy, str) else "balanced"
                    policy = TCQPolicy(
                        num_layers=num_layers, base_format="turbo4", vbr_policy=vbr_pol
                    )
            else:
                policy = None
            self.tcq_policy = policy

            self._pool = TurboMHAKVCache(
                num_kv_heads=num_kv_heads,
                num_layers=num_layers,
                head_dim=head_dim,
                num_pages=num_pages,
                page_size=page_size,
                dtype=dtype,
                device=device,
                layer_ids=layer_ids,
                book="turbo4" if kv_format in ["vbr", "tcq"] else kv_format,
                policy=policy,
            )
        else:
            self.tcq_policy = None
            from .mha_pool import MHAKVCache

            self._pool = MHAKVCache(
                num_kv_heads=num_kv_heads,
                num_layers=num_layers,
                head_dim=head_dim,
                num_pages=num_pages,
                page_size=page_size,
                dtype=dtype,
                device=device,
                layer_ids=layer_ids,
                host_pages=host_pages,
            )
        self.kv_format = kv_format
        self._mtp_slot: int | None = None
        if mtp_layer_id is not None and layer_ids is not None and mtp_layer_id in layer_ids:
            self._mtp_slot = list(layer_ids).index(mtp_layer_id)
        self._zero_kv_slabs()
        self._alloc_index_tiers(num_pages)
        # One layer's worth of the RAM tier on the device: eager multi-row forwards copy the
        # RAM pages they touch here once per layer instead of re-reading them over PCIe.
        self.host_staging = (
            tuple(
                torch.empty(
                    (host_pages, *self._pool._kv_buffer.shape[3:]), dtype=dtype, device=device
                )
                for _ in range(2)
            )
            if host_pages
            else None
        )

    @property
    def _kv_buffer(self):
        # Keep the pool's established inspection surface while storage is delegated.
        return self._pool._kv_buffer

    @_kv_buffer.setter
    def _kv_buffer(self, value):
        self._pool._kv_buffer = value

    def _zero_kv_slabs(self) -> None:
        if hasattr(self._pool, "_kv_buffer"):
            self._pool._kv_buffer.zero_()
        if getattr(self._pool, "_kv_host", None) is not None:
            self._pool._kv_host.zero_()
        if hasattr(self._pool, "_k_codes"):
            self._pool._k_codes.zero_()
            self._pool._k_norm.zero_()
            self._pool._v_codes.zero_()
            self._pool._v_norm.zero_()

    def _alloc_index_tiers(self, num_pages: int) -> None:
        # ZERO-initialized: the score kernel reads whole rows of blocks unmasked and relies on
        # never-written tail rows dotting to a finite 0. Written rows are never cleared again,
        # so the kernel must clamp visible blocks to kvlen // index_ratio.
        self._cmp_scratch_base = num_pages * self._page_size // self._index_ratio
        self._cmp_k_buffer = torch.zeros(
            self._num_index_layers,
            self._cmp_scratch_base + self._num_req_slots,
            self._index_head_dim,
            dtype=self._index_dtype,
            device=self.device,
        )
        self._pending_ring = torch.zeros(
            self._num_req_slots,
            self._num_index_layers,
            self._ring_capacity,
            self._index_head_dim,
            dtype=self._index_dtype,
            device=self.device,
        )
        # 3-axis rope position of every stored token: a compressed group ropes at its first token, which under mrope is not derivable from the logical position
        self._rope_positions = (
            torch.zeros(num_pages * self._page_size, 3, dtype=torch.int32, device=self.device)
            if self._mrope
            else None
        )
        # Hot/cold residency (RAM tier only). The scheduler's page ids are logical: kernels
        # address page_map[logical], and rebalance() swaps page contents between the tiers
        # stream-ordered, so the page table, radix tree and free list never see a move.
        if getattr(self._pool, "_host_pages", 0):
            ids = torch.arange(num_pages, dtype=torch.int32, device=self.device)
            self.page_map = ids
            self._page_owner = ids.clone()
            # Selections per physical page, halved every rebalance; the last entry sinks misses.
            self.page_heat = torch.zeros(num_pages + 1, dtype=torch.int32, device=self.device)
        else:
            self.page_map = None

    def rebuild(self, num_pages: int) -> None:
        # Free the index tiers BEFORE the K/V realloc (super().rebuild frees + syncs +
        # empty_cache), then re-derive them at the new page count. If the index alloc itself
        # fails (OOM), null the K/V slab too and re-raise: a pool with a grown K/V slab and no
        # index slab would mis-serve silently. Rebuild is idle-only, so zeroing the ring here
        # cannot drop a live request's pending members.
        self._cmp_k_buffer = None
        self._pending_ring = None
        self._rope_positions = None
        self._pool.rebuild(num_pages)
        self._zero_kv_slabs()
        try:
            self._alloc_index_tiers(num_pages)
        except Exception:
            if hasattr(self._pool, "_kv_buffer"):
                self._pool._kv_buffer = None
                self._pool._k_buffer = None
                self._pool._v_buffer = None
            raise

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token
        from freetoken.attention import AttnType

        num_req_slots = config.max_running_req + 1
        per_token = 0
        fixed = 0
        for spec in config.model_config.kv_cache_group_specs():
            if spec.is_swa:
                continue
            per_token += spec_kv_bytes_per_token(spec, config)
            if spec.attn_type is AttnType.QSA:
                # One index-key row = all index layers at one position.
                row = spec.index_head_dim * spec.num_index_layers * _INDEX_DTYPE_BYTES
                spec_mtp = getattr(config, "spec_mtp", 0)
                fixed += (
                    num_req_slots * row * (cls.ring_capacity_for(spec.index_ratio, spec_mtp) + 1)
                )
                if config.model_config.model_is_mrope:
                    per_token += _ROPE_POS_BYTES
        return per_token * config.page_size, fixed, config.page_size, 0

    @classmethod
    def host_tier_device_bytes(cls, config, host_tokens: int) -> int:
        """Device (VRAM) bytes the RAM tier still costs per host token even though its K/V
        pages live in pinned host RAM: the compressed index row and rope row (``_cmp_k_buffer``
        / ``_rope_positions`` are sized over ALL pages, RAM tier included -- see module
        docstring) plus the ``host_staging`` double buffer (2 tensors of shape
        ``[host_pages, page_size, kv_heads, head_dim]``, bf16)."""
        from freetoken.attention import AttnType
        from freetoken.utils import div_even

        per_token = 0
        for spec in config.model_config.kv_cache_group_specs():
            if spec.is_swa or spec.attn_type is not AttnType.QSA:
                continue
            per_token += (
                spec.index_head_dim * spec.num_index_layers * _INDEX_DTYPE_BYTES
            ) // spec.index_ratio
            if config.model_config.model_is_mrope:
                per_token += _ROPE_POS_BYTES
            local_kv_heads = div_even(spec.num_kv_heads, config.tp_info.size, allow_replicate=True)
            per_token += 2 * local_kv_heads * spec.head_dim * config.dtype.itemsize
        return per_token * host_tokens

    def unit_bytes(self) -> tuple[int, int]:
        kv, swa = self._pool.unit_bytes()
        if hasattr(self._pool, "_kv_buffer"):
            tokens = int(self._pool._kv_buffer.shape[2]) * int(self._pool._kv_buffer.shape[3])
        else:
            tokens = int(self._pool._num_storage_layers) * self._pool._tokens
        slab = (
            self._num_index_layers
            * self._cmp_scratch_base
            * self._index_head_dim
            * self._index_dtype.itemsize
        )
        return kv + slab // tokens + (_ROPE_POS_BYTES if self._mrope else 0), swa

    def rebuild_from_config(self, config, num_pages: int, **kwargs) -> None:
        self.rebuild(num_pages + 1)

    @property
    def device(self) -> torch.device:
        return self._pool.device

    @property
    def dtype(self) -> torch.dtype:
        return self._pool.dtype

    @property
    def num_layers(self) -> int:
        return self._pool.num_layers

    def k_cache(self, index: int) -> torch.Tensor:
        if getattr(self._pool, "compressed", False):
            return self._pool._k_codes[self._pool._dense(index)]
        return self._pool.k_cache(index)

    def v_cache(self, index: int) -> torch.Tensor:
        if getattr(self._pool, "compressed", False):
            return self._pool._v_codes[self._pool._dense(index)]
        return self._pool.v_cache(index)

    def k_slab(self, index: int):
        return self._pool.k_slab(index)

    def v_slab(self, index: int):
        return self._pool.v_slab(index)

    @property
    def compressed(self) -> bool:
        return getattr(self._pool, "compressed", False)

    @property
    def book3(self) -> bool:
        return getattr(self._pool, "book3", False)

    def is_book3(self, layer_id: int, side: str = "k") -> bool:
        return getattr(self._pool, "is_book3", lambda lid, s="k": self.book3)(layer_id, side)

    @property
    def cent_tensor(self) -> torch.Tensor | None:
        return getattr(self._pool, "cent_tensor", None)

    def store_kv(self, *args, **kwargs) -> None:
        return self._pool.store_kv(*args, **kwargs)

    @property
    def num_device_pages(self) -> int:
        """Physical pages below this id are on the device; the rest are in the RAM tier."""
        return getattr(self._pool, "num_device_pages", self._cmp_scratch_base)

    def host_kv(self, index: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        host_kv = getattr(self._pool, "host_kv", None)
        return None if host_kv is None else host_kv(index)

    def rebalance(self, max_swaps: int) -> None:
        """Swap up to ``max_swaps`` of the hottest RAM pages with the coldest device pages.

        Pure device work on the current stream (no host sync): every kernel enqueued after this
        reads the new placement, every kernel enqueued before it read the old one. A pair swaps
        only when the RAM page was selected more than twice as often (hysteresis)."""
        from freetoken.kernel.triton.qsa.tiered import swap_pages

        device_pages = self.num_device_pages
        total = self.page_map.shape[0]
        k = min(max_swaps, device_pages, total - device_pages)
        if k <= 0:
            return
        heat = self.page_heat[:total]
        hot, host = torch.topk(heat[device_pages:], k)
        cold, dev = torch.topk(heat[:device_pages], k, largest=False)
        swap = hot > 2 * cold + 1
        active = swap.to(torch.int32)
        phys_host = host + device_pages
        pool = self._pool
        swap_pages(pool._kv_buffer.flatten(0, 1), pool._kv_host.flatten(0, 1), dev, host, active)
        per_page = self._page_size // self._index_ratio
        cmp = self._cmp_k_buffer[:, : total * per_page].unflatten(1, (total, per_page))
        swap_pages(cmp, cmp, dev, phys_host, active)
        if self._rope_positions is not None:
            rope = self._rope_positions.view(1, total, self._page_size, 3)
            swap_pages(rope, rope, dev, phys_host, active)
        heat_dev, heat_host = heat[dev], heat[phys_host]
        heat[dev] = torch.where(swap, heat_host, heat_dev)
        heat[phys_host] = torch.where(swap, heat_dev, heat_host)
        owner_dev = self._page_owner[dev].long()
        owner_host = self._page_owner[phys_host].long()
        self.page_map[owner_dev] = torch.where(swap, phys_host, dev).to(torch.int32)
        self.page_map[owner_host] = torch.where(swap, dev, phys_host).to(torch.int32)
        self._page_owner[dev] = torch.where(swap, owner_host, owner_dev).to(torch.int32)
        self._page_owner[phys_host] = torch.where(swap, owner_dev, owner_host).to(torch.int32)
        self.page_heat.bitwise_right_shift_(1)

    def cmp_k_cache(self, slot: int) -> torch.Tensor:
        """Compressed index keys of one sparse layer: ``[rows, index_head_dim]``."""
        return self._cmp_k_buffer[slot]

    def pending_ring(self, slot: int) -> torch.Tensor:
        """One sparse layer's pending ring: ``[num_req_slots, ring_capacity, index_head_dim]``."""
        return self._pending_ring[:, slot]

    @property
    def rope_positions(self) -> torch.Tensor:
        """``[num_tokens, 3]`` int32 t/h/w rope position per KV slot (written by the QSA backend)."""
        assert self._rope_positions is not None, "rope positions are only kept on mrope models"
        return self._rope_positions

    @property
    def cmp_scratch_base(self) -> int:
        """First scratch row of ``cmp_k_cache``; row ``cmp_scratch_base + table_idx`` sinks a
        forward whose group does not close."""
        return self._cmp_scratch_base

    @property
    def index_ratio(self) -> int:
        return self._index_ratio

    @property
    def index_head_dim(self) -> int:
        return self._index_head_dim

    @property
    def ring_capacity(self) -> int:
        return self._ring_capacity

    @property
    def num_req_slots(self) -> int:
        return self._num_req_slots

    def clear_mtp_slot(self) -> None:
        """Zero the MTP draft layer KV slab and compressed index buffer to avoid carrier leak."""
        slot = self._mtp_slot
        if slot is None:
            return
        if self._cmp_k_buffer is not None:
            self._cmp_k_buffer[slot].zero_()
        if self._pending_ring is not None:
            self._pending_ring[:, slot].zero_()
        if hasattr(self._pool, "_k_codes"):
            self._pool._k_codes[slot].zero_()
            self._pool._k_norm[slot].zero_()
            self._pool._v_codes[slot].zero_()
            self._pool._v_norm[slot].zero_()
        elif hasattr(self._pool, "_kv_buffer") and self._pool._kv_buffer is not None:
            self._pool._kv_buffer[:, slot].zero_()
            if getattr(self._pool, "_kv_host", None) is not None:
                from freetoken.kernel.triton.qsa.tiered import zero_tier

                zero_tier(self._pool._kv_host[0, slot])
                zero_tier(self._pool._kv_host[1, slot])

    def free_req(self, table_idx: int) -> None:
        """Zero the per-request pending ring and scratch cmp buffer when table_idx is released."""
        if self._pending_ring is not None and 0 <= table_idx < self._num_req_slots:
            self._pending_ring[table_idx].zero_()
        if self._cmp_k_buffer is not None and 0 <= table_idx < self._num_req_slots:
            self._cmp_k_buffer[:, self._cmp_scratch_base + table_idx].zero_()
        self.clear_mtp_slot()


__all__ = ["QSAKVCache"]
