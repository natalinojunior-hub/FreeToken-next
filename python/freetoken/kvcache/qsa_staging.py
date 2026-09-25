"""Logical QSA page mapping for the opt-in KV-RAM path.

The normal QSA pool still owns its all-VRAM buffers.  This adapter only
produces stable metadata for a staged execution; kernels never receive host
addresses.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
import torch


@dataclass(frozen=True)
class QSAStageEntry:
    """One logical page and its staged device slot."""

    page_id: int
    request_id: int
    logical_position: int
    generation: int
    device_slot: int
    index_group: int
    rope_position: int


class QSAStagingAdapter:
    """Build QSA block and pending-ring metadata from logical pages.

    ``page_size`` is measured in tokens.  A page maps to one contiguous range
    of index groups because QSA requires page-size divisibility by ratio.
    """

    def __init__(self, page_size: int, index_ratio: int, ring_capacity: int) -> None:
        if page_size <= 0 or index_ratio <= 0 or page_size % index_ratio:
            raise ValueError("QSA page_size must be a positive multiple of index_ratio")
        if ring_capacity < index_ratio:
            raise ValueError("QSA ring_capacity must cover one index group")
        self.page_size = page_size
        self.index_ratio = index_ratio
        self.ring_capacity = ring_capacity

    def entry(
        self,
        *,
        page_id: int,
        request_id: int,
        logical_position: int,
        generation: int,
        device_slot: int,
    ) -> QSAStageEntry:
        if min(page_id, request_id, logical_position, generation, device_slot) < 0:
            raise ValueError("QSA stage metadata must be non-negative")
        if logical_position % self.page_size:
            raise ValueError("QSA logical pages must start at a page boundary")
        return QSAStageEntry(
            page_id,
            request_id,
            logical_position,
            generation,
            device_slot,
            logical_position // self.index_ratio,
            logical_position,
        )

    def block_table(self, entries: list[QSAStageEntry]) -> tuple[int, ...]:
        """Return the stable device-slot table indexed by logical page id."""
        if not entries:
            return ()
        by_id: dict[int, QSAStageEntry] = {}
        for entry in entries:
            old = by_id.get(entry.page_id)
            if old is not None and old != entry:
                raise ValueError(f"conflicting QSA page {entry.page_id}")
            by_id[entry.page_id] = entry
        return tuple(by_id[i].device_slot for i in sorted(by_id))

    def ring_slot(self, request_id: int, logical_position: int) -> int:
        if request_id < 0 or logical_position < 0:
            raise ValueError("QSA ring coordinates must be non-negative")
        return logical_position % self.ring_capacity

    def validate_generation(
        self, entry: QSAStageEntry, generation: int, *, device_slot: int | None = None
    ) -> None:
        """Reject stale remaps before a kernel can consume them."""
        if entry.generation != generation:
            raise RuntimeError(f"stale QSA page {entry.page_id}")
        if device_slot is not None and entry.device_slot != device_slot:
            raise RuntimeError(f"QSA page {entry.page_id} moved during staging")

    def remap_block_table(
        self,
        logical_table: torch.Tensor,
        entries: Iterable[QSAStageEntry],
        *,
        generations: dict[int, int] | None = None,
    ) -> torch.Tensor:
        """Materialize a device page table from logical QSA page ids.

        The result is a fresh device tensor, so kernels never receive a host
        pointer or a mutable view of the scheduler's physical table. ``-1``
        remains the invalid-page sentinel used by the QSA kernels.
        """
        if logical_table.ndim != 2 or logical_table.dtype != torch.int32:
            raise ValueError("QSA logical block table must be a 2-D int32 tensor")
        if not logical_table.is_cuda:
            raise ValueError("QSA staged block table must reside on CUDA")
        by_id = {entry.page_id: entry for entry in entries}
        mapped = logical_table.detach().clone()
        for page_id, entry in by_id.items():
            if generations is not None:
                self.validate_generation(entry, generations.get(page_id, -1))
            mapped[logical_table == page_id] = entry.device_slot
        unknown = logical_table >= 0
        for page_id in torch.unique(logical_table[unknown]).tolist():
            if page_id not in by_id:
                raise RuntimeError(f"QSA logical page {page_id} has no staged slot")
        return mapped.contiguous()


__all__ = ["QSAStageEntry", "QSAStagingAdapter"]
