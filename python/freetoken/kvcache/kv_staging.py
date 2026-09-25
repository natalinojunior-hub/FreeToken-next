"""Backend-neutral KV staging metadata.

The adapter only describes and validates a device staging slot.  Existing
all-VRAM pools keep their current physical buffers; callers opt into this
module when a page has already passed the ``KVPagePool`` residency checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import torch

from .kv_tiering import KVPageRecord, KVLayout, Residency


@dataclass(frozen=True)
class StagedKV:
    """Stable metadata consumed by a graph or eager backend invocation."""

    backend: str
    slot: int
    position: int
    generation: int
    k_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    k_stride: tuple[int, ...]
    v_stride: tuple[int, ...]
    qsa_group: int | None


class KVBackendAdapter:
    """Validate a logical page before exposing a backend staging descriptor."""

    backend: str

    def __init__(self, backend: str) -> None:
        if backend not in {"mha", "turbo3", "turbo4"}:
            raise ValueError(f"unsupported staging backend {backend!r}")
        self.backend = backend

    def describe(self, record: KVPageRecord, slot: int) -> StagedKV:
        record.validate()
        if record.layout.format != self.backend:
            raise ValueError(f"layout {record.layout.format!r} does not match {self.backend!r}")
        if record.residency is not Residency.RESIDENT:
            raise RuntimeError("cannot stage a non-resident KV page")
        if slot < 0:
            raise ValueError("staging slot must be non-negative")
        layout = record.layout
        return StagedKV(
            self.backend,
            slot,
            layout.position,
            record.generation,
            layout.k_shape,
            layout.v_shape,
            layout.k_stride,
            layout.v_stride,
            record.qsa_group,
        )

    @staticmethod
    def validate_tensor(
        tensor: torch.Tensor, shape: tuple[int, ...], stride: tuple[int, ...]
    ) -> None:
        if tuple(tensor.shape) != shape or tuple(tensor.stride()) != stride:
            raise ValueError(
                f"staging tensor mismatch: shape={tuple(tensor.shape)} stride={tuple(tensor.stride())}"
            )

    def stage(self, source: torch.Tensor, destination: torch.Tensor, layout: KVLayout) -> None:
        """Copy one already-resident page into a preallocated stable buffer."""
        layout.validate()
        if layout.format != self.backend:
            raise ValueError("layout/backend mismatch")
        self.validate_tensor(source, layout.k_shape, layout.k_stride)
        self.validate_tensor(destination, layout.k_shape, layout.k_stride)
        destination.copy_(source, non_blocking=destination.is_cuda)

    def stage_pair(
        self,
        source_k: torch.Tensor,
        source_v: torch.Tensor,
        destination_k: torch.Tensor,
        destination_v: torch.Tensor,
        layout: KVLayout,
    ) -> None:
        """Stage K and V together, validating each backend layout independently."""
        layout.validate()
        if layout.format != self.backend:
            raise ValueError("layout/backend mismatch")
        self.validate_tensor(source_k, layout.k_shape, layout.k_stride)
        self.validate_tensor(destination_k, layout.k_shape, layout.k_stride)
        self.validate_tensor(source_v, layout.v_shape, layout.v_stride)
        self.validate_tensor(destination_v, layout.v_shape, layout.v_stride)
        destination_k.copy_(source_k, non_blocking=destination_k.is_cuda)
        destination_v.copy_(source_v, non_blocking=destination_v.is_cuda)

    @staticmethod
    def validate_norms(norms: torch.Tensor, tokens: int, heads: int, groups: int) -> None:
        """Turbo norm slabs have one fp16 scale per token/head/group."""
        expected = (tokens, heads, groups)
        if tuple(norms.shape) != expected:
            raise ValueError(f"norm slab mismatch: shape={tuple(norms.shape)} expected={expected}")


class KVStagingBinding:
    """Stable device indirection consumed by live attention launchers.

    The logical page id remains owned by the scheduler.  Kernels receive this
    preallocated device table, whose entries are physical pool slots.  Updating
    entries in place keeps CUDA graph addresses stable; callers must update it
    only before the corresponding graph replay or eager launch.
    """

    def __init__(self, capacity: int, device: torch.device) -> None:
        if capacity <= 0:
            raise ValueError("staging table capacity must be positive")
        self.table = torch.full((capacity,), -1, dtype=torch.int32, device=device)
        self._generation = torch.full((capacity,), -1, dtype=torch.int64, device=device)

    @property
    def device(self) -> torch.device:
        return self.table.device

    def update(
        self, page_ids: Iterable[int], slots: Iterable[int], generations: Iterable[int]
    ) -> None:
        ids = tuple(int(v) for v in page_ids)
        physical = tuple(int(v) for v in slots)
        versions = tuple(int(v) for v in generations)
        if not (len(ids) == len(physical) == len(versions)):
            raise ValueError("staging update vectors must have equal length")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate logical page id in staging update")
        if any(i < 0 or i >= self.table.numel() for i in ids):
            raise IndexError("logical page id outside staging table")
        if any(s < 0 for s in physical) or any(g < 0 for g in versions):
            raise ValueError("physical slots and generations must be non-negative")
        if ids:
            index = torch.tensor(ids, dtype=torch.long, device=self.device)
            self.table.index_copy_(
                0, index, torch.tensor(physical, dtype=torch.int32, device=self.device)
            )
            self._generation.index_copy_(
                0, index, torch.tensor(versions, dtype=torch.int64, device=self.device)
            )

    def clear(self, page_ids: Iterable[int]) -> None:
        ids = tuple(int(v) for v in page_ids)
        if any(i < 0 or i >= self.table.numel() for i in ids):
            raise IndexError("logical page id outside staging table")
        if ids:
            index = torch.tensor(ids, dtype=torch.long, device=self.device)
            self.table.index_fill_(0, index, -1)
            self._generation.index_fill_(0, index, -1)

    def validate(self, page_ids: Iterable[int], generations: Iterable[int]) -> None:
        ids = tuple(int(v) for v in page_ids)
        versions = tuple(int(v) for v in generations)
        if len(ids) != len(versions):
            raise ValueError("validation vectors must have equal length")
        if any(i < 0 or i >= self.table.numel() for i in ids):
            raise IndexError("logical page id outside staging table")
        if ids:
            index = torch.tensor(ids, dtype=torch.long, device=self.device)
            slots = self.table.index_select(0, index)
            current = self._generation.index_select(0, index)
            if bool((slots < 0).any()) or bool(
                (current != torch.tensor(versions, dtype=torch.int64, device=self.device)).any()
            ):
                raise RuntimeError("staging table contains cold or stale KV page")

    @staticmethod
    def validate_norms(norms: torch.Tensor, tokens: int, heads: int, groups: int) -> None:
        """Turbo norm slabs are one fp16 scale per token/head/group."""
        if tuple(norms.shape) != (tokens, heads, groups):
            raise ValueError(
                f"norm slab mismatch: shape={tuple(norms.shape)} expected={(tokens, heads, groups)}"
            )


MHAKVAdapter = KVBackendAdapter("mha")
Turbo3KVAdapter = KVBackendAdapter("turbo3")
Turbo4KVAdapter = KVBackendAdapter("turbo4")
