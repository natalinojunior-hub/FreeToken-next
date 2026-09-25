from dataclasses import replace

import pytest
import torch

from freetoken.kvcache.kv_staging import (
    KVBackendAdapter,
    KVStagingBinding,
    MHAKVAdapter,
    Turbo3KVAdapter,
    Turbo4KVAdapter,
)


def test_staging_binding_updates_in_place_and_rejects_stale_generation():
    binding = KVStagingBinding(4, torch.device("cpu"))
    address = binding.table.data_ptr()
    binding.update([1, 2], [7, 8], [3, 4])
    assert binding.table.data_ptr() == address
    binding.validate([1, 2], [3, 4])
    with pytest.raises(RuntimeError, match="stale"):
        binding.validate([1], [9])
    binding.clear([1])
    with pytest.raises(RuntimeError, match="cold"):
        binding.validate([1], [3])


def test_staging_binding_rejects_duplicate_or_mismatched_updates():
    binding = KVStagingBinding(2, torch.device("cpu"))
    with pytest.raises(ValueError, match="duplicate"):
        binding.update([0, 0], [1, 2], [1, 1])
    with pytest.raises(ValueError, match="equal length"):
        binding.update([0], [1], [])
from freetoken.kvcache.kv_tiering import KVLayout, KVPageRecord, Residency


def record(fmt: str = "mha") -> KVPageRecord:
    layout = KVLayout(fmt, (2, 3, 4), (2, 3, 4), (12, 4, 1), (12, 4, 1), 16)
    return KVPageRecord(
        1, 2, 16, 4, layout, (0, 1, 2), 1, qsa_group=None, residency=Residency.RESIDENT
    )


def test_describe_preserves_slot_position_and_generation():
    desc = MHAKVAdapter.describe(record(), 7)
    assert (desc.slot, desc.position, desc.generation) == (7, 16, 4)
    assert desc.k_stride == (12, 4, 1)


@pytest.mark.parametrize("adapter", [Turbo3KVAdapter, Turbo4KVAdapter])
def test_turbo_layout_and_norm_shape(adapter):
    desc = adapter.describe(record(adapter.backend), 1)
    assert desc.backend == adapter.backend
    adapter.validate_norms(torch.zeros(2, 3, 1), 2, 3, 1)


def test_adapter_rejects_cold_or_wrong_format():
    with pytest.raises(RuntimeError, match="non-resident"):
        MHAKVAdapter.describe(replace(record(), residency=Residency.COLD), 0)
    with pytest.raises(ValueError, match="does not match"):
        MHAKVAdapter.describe(record("turbo3"), 0)


def test_stage_checks_layout_and_strides():
    adapter = KVBackendAdapter("mha")
    layout = record().layout
    src = torch.arange(24).reshape(2, 3, 4)
    dst = torch.zeros_like(src)
    adapter.stage(src, dst, layout)
    assert torch.equal(src, dst)
    with pytest.raises(ValueError, match="staging tensor mismatch"):
        adapter.stage(src.transpose(0, 1), dst, layout)


def test_stage_pair_validates_k_and_v_layouts():
    adapter = KVBackendAdapter("mha")
    layout = record().layout
    k = torch.ones(layout.k_shape)
    v = torch.full(layout.v_shape, 2)
    out_k = torch.zeros_like(k)
    out_v = torch.zeros_like(v)
    adapter.stage_pair(k, v, out_k, out_v, layout)
    assert torch.equal(k, out_k) and torch.equal(v, out_v)


def test_norm_shape_is_strict():
    with pytest.raises(ValueError, match="norm slab mismatch"):
        Turbo4KVAdapter.validate_norms(torch.zeros(2, 3), 2, 3, 1)
