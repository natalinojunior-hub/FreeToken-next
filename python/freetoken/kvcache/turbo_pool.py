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


class LayeredCodes(list):
    """Holds per-layer compressed KV codes allowing mixed-tier (turbo3/turbo4) storage."""

    def zero_(self):
        for t in self:
            t.zero_()
        return self

    def numel(self) -> int:
        return sum(t.numel() for t in self)


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
        policy=None,
    ) -> None:
        from freetoken.kernel.triton.turbo_kv import CODE_BYTES, QK_TURBO, BOOKS

        if book in ("vbr", "tcq") and policy is None:
            from freetoken.kvcache.tcq_policy import TCQPolicy

            policy = TCQPolicy(num_layers, base_format="turbo4", vbr_policy="balanced")

        if book not in BOOKS and book not in ("vbr", "tcq"):
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
        if book == "nvfp4" and (
            device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 10
        ):
            raise ValueError(
                "--kv-format nvfp4 needs a Blackwell GPU (sm_100+/sm_120): the attention kernel "
                "decodes e2m1 with the hardware F2FP.E2M1 conversion"
            )
        self.book = "turbo4" if book in ("vbr", "tcq") else book
        self.policy = policy
        self._dtype = dtype
        self._device = device
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._page_size = page_size
        self._groups = head_dim // QK_TURBO
        self._code_bytes = CODE_BYTES[self.book]
        self._layer_map: list[int] | None = None
        self._storage_layer_ids = (
            list(layer_ids) if layer_ids is not None else list(range(num_layers))
        )
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
        from freetoken.kernel.triton.turbo_kv import CENTROIDS_3, CENTROIDS_4

        self._cent_3 = torch.tensor(CENTROIDS_3, device=device, dtype=torch.float32)
        self._cent_4 = torch.tensor(CENTROIDS_4, device=device, dtype=torch.float32)
        self._cent = self._cent_3 if self.book == "turbo3" else self._cent_4

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def book3(self) -> bool:
        return self.book == "turbo3"

    def get_tier(self, layer_id: int, side: str = "k") -> str:
        if self.policy is not None:
            return self.policy.get_tier(layer_id, side)
        return self.book

    def is_book3(self, layer_id: int, side: str = "k") -> bool:
        return self.get_tier(layer_id, side) == "turbo3"

    @property
    def cent_tensor(self) -> torch.Tensor:
        """The lookup book the tile readers index; a device constant, not per-call state."""
        return self._cent

    def _alloc(self, num_pages: int) -> None:
        from freetoken.kernel.triton.turbo_kv import CODE_BYTES

        tokens = num_pages * self._page_size
        self._tokens = tokens
        nshape = (self._num_storage_layers, tokens, self._num_kv_heads, self._groups)
        self._k_norm = torch.zeros(nshape, device=self._device, dtype=torch.float16)
        self._v_norm = torch.zeros(nshape, device=self._device, dtype=torch.float16)

        if self.policy is not None and getattr(self.policy, "has_mixed_tiers", True):
            k_list, v_list = [], []
            for dense in range(self._num_storage_layers):
                gid = self._storage_layer_ids[dense]
                kt = self.get_tier(gid, "k")
                vt = self.get_tier(gid, "v")
                kb = CODE_BYTES[kt]
                vb = CODE_BYTES[vt]
                k_list.append(
                    torch.zeros(
                        (tokens, self._num_kv_heads, self._groups * kb),
                        device=self._device,
                        dtype=torch.uint8,
                    )
                )
                v_list.append(
                    torch.zeros(
                        (tokens, self._num_kv_heads, self._groups * vb),
                        device=self._device,
                        dtype=torch.uint8,
                    )
                )
            self._k_codes = LayeredCodes(k_list)
            self._v_codes = LayeredCodes(v_list)
        else:
            shape = (
                self._num_storage_layers,
                tokens,
                self._num_kv_heads,
                self._groups * self._code_bytes,
            )
            self._k_codes = torch.zeros(shape, device=self._device, dtype=torch.uint8)
            self._v_codes = torch.zeros(shape, device=self._device, dtype=torch.uint8)

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

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from freetoken.kernel.triton.turbo_kv import quantize, CODE_BYTES

        dense = self._dense(layer_id)
        rows = out_loc.numel()
        # the page table hands us int32 slots and index_copy_ demands int64
        slots = out_loc if out_loc.dtype is torch.long else out_loc.long()
        k_book = self.get_tier(layer_id, "k")
        v_book = self.get_tier(layer_id, "v")
        k_stride = CODE_BYTES[k_book] * self._groups
        v_stride = CODE_BYTES[v_book] * self._groups
        k2 = k.reshape(rows, self._num_kv_heads, self._head_dim)
        v2 = v.reshape(rows, self._num_kv_heads, self._head_dim)
        kq, kn = quantize(k2.reshape(-1, self._head_dim).to(self._dtype), k_book)
        vq, vn = quantize(v2.reshape(-1, self._head_dim).to(self._dtype), v_book)
        self._k_codes[dense].index_copy_(0, slots, kq.reshape(rows, self._num_kv_heads, k_stride))
        self._k_norm[dense].index_copy_(
            0, slots, kn.reshape(rows, self._num_kv_heads, self._groups)
        )
        self._v_codes[dense].index_copy_(0, slots, vq.reshape(rows, self._num_kv_heads, v_stride))
        self._v_norm[dense].index_copy_(
            0, slots, vn.reshape(rows, self._num_kv_heads, self._groups)
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

    def decode_rows(self, layer_id: int, which: str = "k") -> torch.Tensor:
        """Materialize the whole slab. Only for tests and the measurement arm that quantifies how
        much the fused path saves -- never on a serving hot path."""
        from freetoken.kernel.triton.turbo_kv import decode_rotated, inv_rotate, CODE_BYTES

        codes, norm = (
            (self._k_codes, self._k_norm) if which == "k" else (self._v_codes, self._v_norm)
        )
        dense = self._dense(layer_id)
        tokens, heads = (
            codes.shape[1] if hasattr(codes, "shape") else codes[dense].shape[0],
            codes.shape[2] if hasattr(codes, "shape") else codes[dense].shape[1],
        )
        book = self.get_tier(layer_id, which)
        code_b = CODE_BYTES[book]
        flat = codes[dense].reshape(-1, code_b)
        nflat = norm[dense].reshape(-1, 1)
        y = decode_rotated(flat, nflat, book)
        return inv_rotate(y).reshape(tokens, heads, self._head_dim).to(self._dtype)

    # ---- sizing ------------------------------------------------------------------

    @classmethod
    def kv_cost(cls, config, **kwargs) -> tuple[int, int, int, int]:
        book = kwargs.get("book") or getattr(config, "kv_format", "auto")
        if book not in ("turbo3", "turbo4", "fp8", "nvfp4"):
            raise ValueError(f"TurboMHAKVCache priced with book {book!r}")
        per_token = 0
        for spec in config.model_config.kv_cache_group_specs():
            if spec.is_swa:
                continue
            heads = _local_heads(spec, config)
            per_token += packed_bytes_per_token(spec.head_dim, heads, book) * spec.num_layers
        return per_token * config.page_size, 0, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
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
