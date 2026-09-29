"""CUDA virtual memory for the expert slot arenas: a fixed virtual range whose physical backing
can grow and shrink in place. Addresses never move, so resident slots stay warm and captured
CUDA graphs stay valid across a resize; only the backed prefix of each range is touchable."""

from __future__ import annotations

import ctypes

import torch

_CU_MEM_ALLOCATION_TYPE_PINNED = 1
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
_CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1


class _Location(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _AllocProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _Location),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class _AccessDesc(ctypes.Structure):
    _fields_ = [("location", _Location), ("flags", ctypes.c_int)]


_lib = None


class VMMResizeRollbackError(RuntimeError):
    """A failed resize could not restore its mapped ranges to a coherent prefix."""


def _cu():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL("libcuda.so.1")
    return _lib


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed with CUresult {rc}")


def _prop(device: int) -> _AllocProp:
    prop = _AllocProp()
    prop.type = _CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location = _Location(_CU_MEM_LOCATION_TYPE_DEVICE, device)
    return prop


def granularity(device: int) -> int:
    g = ctypes.c_size_t()
    prop = _prop(device)
    _check(
        _cu().cuMemGetAllocationGranularity(
            ctypes.byref(g), ctypes.byref(prop), _CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
        ),
        "cuMemGetAllocationGranularity",
    )
    return g.value


def supported(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    try:
        return granularity(device.index or 0) > 0
    except (OSError, RuntimeError):
        return False


class _CudaArray:
    """``__cuda_array_interface__`` over a raw device pointer, for ``torch.as_tensor``."""

    def __init__(self, ptr: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
            "strides": None,
            "stream": None,
        }


class VirtualArena:
    """One reserved virtual range of ``nbytes``; ``regions`` are disjoint granule-aligned
    ``(offset, max_bytes)`` spans, each backed from its start up to ``backed[i]`` bytes."""

    def __init__(self, device: torch.device, nbytes: int, regions: list[tuple[int, int]]) -> None:
        self.device_id = device.index if device.index is not None else torch.cuda.current_device()
        self.g = granularity(self.device_id)
        self.nbytes = -(-nbytes // self.g) * self.g
        ptr = ctypes.c_uint64()
        _check(
            _cu().cuMemAddressReserve(ctypes.byref(ptr), ctypes.c_size_t(self.nbytes), 0, 0, 0),
            "cuMemAddressReserve",
        )
        self.ptr = ptr.value
        self.regions = regions
        self.backed = [0] * len(regions)
        # per region: stack of (offset_in_region, size, handle), mapped in order
        self._chunks: list[list[tuple[int, int, int]]] = [[] for _ in regions]
        self._device = device
        self._tensor: torch.Tensor | None = None

    def _map(self, va: int, size: int) -> int:
        handle = ctypes.c_uint64()
        prop = _prop(self.device_id)
        _check(
            _cu().cuMemCreate(ctypes.byref(handle), ctypes.c_size_t(size), ctypes.byref(prop), 0),
            "cuMemCreate",
        )
        rc = _cu().cuMemMap(
            ctypes.c_uint64(va), ctypes.c_size_t(size), ctypes.c_size_t(0), handle, 0
        )
        if rc != 0:
            _cu().cuMemRelease(handle)
            _check(rc, "cuMemMap")
        access = _AccessDesc(
            _Location(_CU_MEM_LOCATION_TYPE_DEVICE, self.device_id),
            _CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        )
        rc = _cu().cuMemSetAccess(
            ctypes.c_uint64(va), ctypes.c_size_t(size), ctypes.byref(access), 1
        )
        if rc != 0:
            _cu().cuMemUnmap(ctypes.c_uint64(va), ctypes.c_size_t(size))
            _cu().cuMemRelease(handle)
            _check(rc, "cuMemSetAccess")
        return handle.value

    def set_backed(self, region: int, nbytes: int) -> int:
        """Back exactly ``region``'s first ``nbytes`` (granule-rounded) and return how many of
        them kept their contents. Physical handles cover one native granule each, so shrinking
        only unmaps the tail and keeps the entire retained prefix warm without allocating.
        The caller has synchronized every kernel that could touch the range."""
        off, cap = self.regions[region]
        want = min(cap, -(-nbytes // self.g) * self.g)
        chunks = self._chunks[region]
        intact = self.backed[region]
        while chunks and chunks[-1][0] + chunks[-1][1] > want:
            c_off, size, handle = chunks[-1]
            _check(
                _cu().cuMemUnmap(ctypes.c_uint64(self.ptr + off + c_off), ctypes.c_size_t(size)),
                "cuMemUnmap",
            )
            chunks.pop()
            self.backed[region] = c_off
            try:
                _check(_cu().cuMemRelease(ctypes.c_uint64(handle)), "cuMemRelease")
            except Exception as error:
                raise VMMResizeRollbackError(
                    f"VMM shrink unmapped through {c_off} but could not release a handle: {error!r}"
                ) from error
            intact = min(intact, c_off)
        have = chunks[-1][0] + chunks[-1][1] if chunks else 0
        original_chunks = len(chunks)
        try:
            while want > have:
                chunks.append((have, self.g, self._map(self.ptr + off + have, self.g)))
                have += self.g
        except Exception as error:
            for c_off, size, handle in reversed(chunks[original_chunks:]):
                try:
                    _check(
                        _cu().cuMemUnmap(
                            ctypes.c_uint64(self.ptr + off + c_off), ctypes.c_size_t(size)
                        ),
                        "cuMemUnmap",
                    )
                except Exception as cleanup_error:
                    self.backed[region] = chunks[-1][0] + chunks[-1][1]
                    raise VMMResizeRollbackError(
                        f"VMM growth failed ({error!r}); rollback unmap failed ({cleanup_error!r})"
                    ) from error
                chunks.pop()
                self.backed[region] = chunks[-1][0] + chunks[-1][1] if chunks else 0
                try:
                    _check(_cu().cuMemRelease(ctypes.c_uint64(handle)), "cuMemRelease")
                except Exception as cleanup_error:
                    raise VMMResizeRollbackError(
                        f"VMM growth failed ({error!r}); rollback release failed ({cleanup_error!r})"
                    ) from error
            raise
        self.backed[region] = want
        return min(intact, want)

    @property
    def tensor(self) -> torch.Tensor:
        """uint8 view of the whole range. Built once the start is backed: the driver only
        recognizes the pointer as device memory where something is mapped."""
        if self._tensor is None:
            assert self._chunks[0] and self.regions[0][0] == 0, "back the first region first"
            self._tensor = torch.as_tensor(_CudaArray(self.ptr, self.nbytes), device=self._device)
        return self._tensor

    @property
    def backed_bytes(self) -> int:
        return sum(self.backed)

    def release(self) -> None:
        """Unmap everything and free the virtual range (no kernel may touch it afterwards)."""
        for r in range(len(self.regions)):
            self.set_backed(r, 0)
        if self.ptr:
            _check(
                _cu().cuMemAddressFree(ctypes.c_uint64(self.ptr), ctypes.c_size_t(self.nbytes)),
                "cuMemAddressFree",
            )
            self.ptr = 0
