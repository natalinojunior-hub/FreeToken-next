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


@dataclass(frozen=True)
class KVLayout:
    """Format metadata required to stage one logical KV page safely."""

    format: str
    k_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    k_stride: tuple[int, ...]
    v_stride: tuple[int, ...]
    position: int

    def validate(self) -> None:
        if self.format not in {"mha", "qsa", "turbo3", "turbo4"}:
            raise ValueError(f"unsupported KV format {self.format!r}")
        if self.position < 0 or not self.k_shape or not self.v_shape:
            raise ValueError("KV layout requires a non-negative position and non-empty shapes")
        if len(self.k_shape) != len(self.k_stride) or len(self.v_shape) != len(self.v_stride):
            raise ValueError("KV layout shape/stride rank mismatch")


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
    prefetch_bytes: int = 0
    eviction_bytes: int = 0
    queue_depth: int = 0


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
    eviction_event: torch.cuda.Event | None = None

    @staticmethod
    def pinned(
        shape: tuple[int, ...], dtype: torch.dtype, page_id: int, logical_position: int
    ) -> "KVPage":
        """Create a page with pinned host backing; fail before admission if unavailable."""
        host = torch.empty(shape, dtype=dtype, pin_memory=True)
        return KVPage(page_id, logical_position, host=host)

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

    def __init__(
        self,
        pages: Iterable[KVPage],
        device_slots: int,
        device: torch.device | str | None = None,
        verify_checksums: bool = False,
    ) -> None:
        if device_slots < 1:
            raise ValueError("device_slots must be positive")
        pages = list(pages)
        if not pages or pages[0].host is None:
            raise ValueError("KV pages require host backing")
        template = pages[0].host
        device = device or (pages[0].device.device if pages[0].device is not None else "cuda")
        for page in pages:
            if (
                page.host is None
                or page.host.shape != template.shape
                or page.host.dtype != template.dtype
            ):
                raise ValueError("KV pages require identical host backing")
            if page.device is not None and page.device.shape != template.shape:
                raise ValueError("host/device page size mismatch")
            page.device = None
        self._slots = [torch.empty_like(template, device=device) for _ in range(device_slots)]
        self.table = PageTable(pages)
        self._free = list(range(device_slots))
        self.telemetry = PageTelemetry()
        self.verify_checksums = verify_checksums

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
        page.device = self._slots[page.device_slot]
        page.residency = Residency.PREFETCHING
        return page

    @staticmethod
    def _checksum(tensor: torch.Tensor) -> int:
        return int(tensor.detach().float().sum().item())

    def prefetch(self, page_id: int, owner: str, stream: torch.cuda.Stream | None = None) -> KVPage:
        page = self.table.get(page_id)
        if page.host is None:
            raise RuntimeError("KV page requires host backing")
        expected = self._checksum(page.host) if self.verify_checksums else None
        if self.verify_checksums and page.checksum is not None and expected != page.checksum:
            raise RuntimeError(f"stale host backing for page {page_id}")
        if self._slots[0].is_cuda and not page.host.is_pinned():
            raise RuntimeError("asynchronous KV prefetch requires pinned host backing")
        page = self.admit(page_id, owner)
        assert page.device is not None
        if page.device.is_cuda and page.host.device.type == "cpu":
            with torch.cuda.stream(stream) if stream is not None else _nullcontext():
                page.device.copy_(page.host, non_blocking=True)
            page.event = torch.cuda.Event()
            page.event.record(stream)
        else:
            page.device.copy_(page.host)
            page.event = None
        page.residency = Residency.RESIDENT
        page.checksum = expected
        self.telemetry.resident += 1
        self.telemetry.prefetch_bytes += page.device.nbytes
        self.telemetry.queue_depth = max(0, self.telemetry.queue_depth - 1)
        return page

    def ready(self, page_id: int) -> bool:
        event = self.table.get(page_id).event
        return event is None or event.query()

    def wait_ready(self, page_id: int, stream: torch.cuda.Stream) -> None:
        event = self.table.get(page_id).event
        if event is not None:
            stream.wait_event(event)

    def evict(self, page_id: int, owner: str, generation: int | None = None) -> bool:
        page = self.table.get(page_id)
        if generation is not None and generation != page.generation:
            return False
        if page.owner != owner:
            return False
        eviction_done = page.residency is Residency.EVICTING
        if eviction_done:
            if page.eviction_event is None or not page.eviction_event.query():
                return False
            page.eviction_event = None
        if page.event is not None and not page.event.query():
            return False
        if page.device is not None and page.host is not None:
            if page.device.is_cuda and not eviction_done:
                if not page.host.is_pinned():
                    return False
                if page.eviction_event is None:
                    page.residency = Residency.EVICTING
                    self.telemetry.queue_depth += 1
                    page.host.copy_(page.device, non_blocking=True)
                    page.eviction_event = torch.cuda.Event()
                    page.eviction_event.record(torch.cuda.current_stream(page.device.device))
                    return False
                if not page.eviction_event.query():
                    return False
                page.eviction_event = None
            else:
                page.host.copy_(page.device)
        if page.device_slot is not None:
            self._free.append(page.device_slot)
        page.device_slot = None
        page.device = None
        page.residency = Residency.COLD
        page.event = None
        page.release(owner)
        if page.host is not None:
            page.checksum = self._checksum(page.host)
        self.telemetry.eviction_bytes += page.device.nbytes if page.device is not None else 0
        self.telemetry.queue_depth = max(0, self.telemetry.queue_depth - 1)
        self.telemetry.resident = max(0, self.telemetry.resident - 1)
        self.telemetry.evictions += 1
        return True

    def progress(self, page_id: int, owner: str, generation: int | None = None) -> bool:
        page = self.table.get(page_id)
        if page.owner != owner or page.residency is not Residency.EVICTING:
            return False
        if generation is not None and generation != page.generation:
            return False
        if page.eviction_event is None or not page.eviction_event.query():
            return False
        return self.evict(page_id, owner, generation)

    def can_replay(self, page_ids: Iterable[int]) -> bool:
        """Graph-safe residency check; false means caller must use eager mode."""
        ok = all(
            self.table.get(page_id).residency is Residency.RESIDENT and self.ready(page_id)
            for page_id in page_ids
        )
        if not ok:
            self.telemetry.rejected_replays += 1
        return ok

    def can_replay_batch(self, batch: object) -> bool:
        """Check an explicitly mapped batch; missing mapping fails closed."""
        page_ids = getattr(batch, "kv_page_ids", None)
        return page_ids is not None and self.can_replay(page_ids)

    def cancel(self, owner: str) -> bool:
        """Begin cleanup and report whether every owned page is released."""
        pending = False
        for page in list(self.table._pages.values()):
            if page.owner == owner and not self.evict(page.page_id, owner):
                pending = True
        return not pending and all(page.owner != owner for page in self.table._pages.values())

    def close(self) -> None:
        """Release an idle pool; refuse shutdown while a request still owns a page."""
        owned = [page.page_id for page in self.table._pages.values() if page.owner is not None]
        if owned:
            raise RuntimeError(f"cannot close KV pool with owned pages: {owned}")
        if any(page.residency is not Residency.COLD for page in self.table._pages.values()):
            raise RuntimeError("cannot close KV pool before asynchronous transitions finish")
        self._slots.clear()
        self._free.clear()


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def tiering_safe(*, cuda_graph: bool, stable_indirection: bool, pinned_host: bool) -> bool:
    """Capability gate; false means retain the all-VRAM path."""
    return cuda_graph and stable_indirection and pinned_host
