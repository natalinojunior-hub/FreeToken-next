"""Backend-neutral KV staging metadata.

The adapter only describes and validates a device staging slot.  Existing
all-VRAM pools keep their current physical buffers; callers opt into this
module when a page has already passed the ``KVPagePool`` residency checks.
"""

from __future__ import annotations

from dataclasses import dataclass

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
