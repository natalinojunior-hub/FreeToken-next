"""Safe KV page ownership and residency state machine.

The manager is deliberately independent of model geometry.  Runtime integration
must opt in only after CUDA graph capability checks; callers can always fall back
to the existing all-device cache without changing page identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import torch


class Residency(str, Enum):
    COLD = "cold"
    PREFETCHING = "prefetching"
    RESIDENT = "resident"
    EVICTING = "evicting"


@dataclass
class PageTelemetry:
    resident: int = 0
    cold_hits: int = 0
    hot_hits: int = 0
    evictions: int = 0
    rejected_replays: int = 0


@dataclass
class KVPage:
    page_id: int
    logical_position: int
    generation: int = 0
    residency: Residency = Residency.COLD
    owner: str | None = None
    checksum: int | None = None
    device_slot: int | None = None
    host: torch.Tensor | None = None
    device: torch.Tensor | None = None
    event: torch.cuda.Event | None = None

    def claim(self, owner: str) -> int:
        if self.owner is not None:
            raise RuntimeError(f"page {self.page_id} owned by {self.owner}")
        self.owner = owner
        self.generation += 1
        return self.generation

    def release(self, owner: str) -> None:
        if self.owner != owner:
            raise RuntimeError(f"page {self.page_id} release by non-owner {owner}")
        self.owner = None
        self.generation += 1


class PageTable:
    """Logical page mapping; entries never expose movable storage addresses."""

    def __init__(self, pages: Iterable[KVPage] = ()) -> None:
        self._pages: dict[int, KVPage] = {p.page_id: p for p in pages}

    def add(self, page: KVPage) -> None:
        if page.page_id in self._pages:
            raise ValueError(f"duplicate page id {page.page_id}")
        self._pages[page.page_id] = page

    def get(self, page_id: int) -> KVPage:
        return self._pages[page_id]

    def generation(self, page_id: int) -> int:
        return self.get(page_id).generation

    def device_slots(self, page_ids: Iterable[int]) -> list[int]:
        out: list[int] = []
        for page_id in page_ids:
            page = self.get(page_id)
            if page.residency is not Residency.RESIDENT or page.device_slot is None:
                raise RuntimeError(f"page {page_id} is not resident")
            out.append(page.device_slot)
        return out


class KVPagePool:
    """Bounded device pool with versioned, idempotent transitions.

    Host tensors are caller-owned pinned copies.  CUDA copies are asynchronous
    when available; a consumer must call ``ready`` before graph/eager use.
    """

    def __init__(self, pages: Iterable[KVPage], device_slots: int) -> None:
        if device_slots < 1:
            raise ValueError("device_slots must be positive")
        self.table = PageTable(pages)
        self._free = list(range(device_slots))
        self.telemetry = PageTelemetry()

    def admit(self, page_id: int, owner: str) -> KVPage:
        page = self.table.get(page_id)
        page.claim(owner)
        if page.residency is Residency.RESIDENT:
            self.telemetry.hot_hits += 1
            return page
        self.telemetry.cold_hits += 1
        if not self._free:
            page.release(owner)
            raise MemoryError("KV device pool exhausted")
        page.device_slot = self._free.pop()
        page.residency = Residency.PREFETCHING
        return page

    def prefetch(self, page_id: int, owner: str, stream: torch.cuda.Stream | None = None) -> KVPage:
        page = self.table.get(page_id)
        if page.host is None or page.device is None:
            raise RuntimeError("KV page requires host and device backing")
        if page.host.numel() != page.device.numel():
            raise ValueError("host/device page size mismatch")
        if page.device.is_cuda and not page.host.is_pinned():
            raise RuntimeError("asynchronous KV prefetch requires pinned host backing")
        page = self.admit(page_id, owner)
        if page.device.is_cuda and page.host.device.type == "cpu":
            with torch.cuda.stream(stream) if stream is not None else _nullcontext():
                page.device.copy_(page.host, non_blocking=True)
            page.event = torch.cuda.Event()
            page.event.record(stream)
        else:
            page.device.copy_(page.host)
            page.event = None
        page.residency = Residency.RESIDENT
        self.telemetry.resident += 1
        return page

    def ready(self, page_id: int) -> bool:
        event = self.table.get(page_id).event
        return event is None or event.query()

    def evict(self, page_id: int, owner: str, generation: int | None = None) -> bool:
        page = self.table.get(page_id)
        if generation is not None and generation != page.generation:
            return False
        if page.owner != owner:
            return False
        if page.event is not None and not page.event.query():
            return False
        if page.device is not None and page.host is not None:
            page.host.copy_(page.device)
        if page.device_slot is not None:
            self._free.append(page.device_slot)
        page.device_slot = None
        page.residency = Residency.COLD
        page.event = None
        page.release(owner)
        self.telemetry.resident = max(0, self.telemetry.resident - 1)
        self.telemetry.evictions += 1
        return True

    def can_replay(self, page_ids: Iterable[int]) -> bool:
        """Graph-safe residency check; false means caller must use eager mode."""
        ok = all(
            self.table.get(page_id).residency is Residency.RESIDENT and self.ready(page_id)
            for page_id in page_ids
        )
        if not ok:
            self.telemetry.rejected_replays += 1
        return ok

    def cancel(self, owner: str) -> None:
        for page in list(self.table._pages.values()):
            if page.owner == owner:
                self.evict(page.page_id, owner)


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def tiering_safe(*, cuda_graph: bool, stable_indirection: bool, pinned_host: bool) -> bool:
    """Capability gate; false means retain the all-VRAM path."""
    return cuda_graph and stable_indirection and pinned_host
