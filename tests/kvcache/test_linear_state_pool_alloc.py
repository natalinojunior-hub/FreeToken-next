"""LinearStatePool unit: the free-list allocator and the declared slot-state siblings.
CPU-only, fast — pure slot bookkeeping + state copy/zero, no kernels."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.kvcache.linear_state_pool import (
    LinearStatePool,
    linear_state_bytes_per_req,
    state_pool_bytes,
)
from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec


def _group():
    return LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=(0, 1),
        num_key_heads=2,
        num_value_heads=4,
        key_head_dim=16,
        value_head_dim=16,
        conv_kernel_dim=4,
        output_gate="silu",
    )


def _pool(num_slots=8, device="cpu", slot_states=()):
    return LinearStatePool(
        group=_group(),
        num_slots=num_slots,
        dtype=torch.bfloat16,
        device=torch.device(device),
        tp_size=1,
        slot_states=slot_states,
    )


def test_alloc_free_roundtrip():
    pool = _pool(num_slots=8)
    assert pool.num_free_slots == 7  # slots 1..7 (slot 0 = padding)
    a = pool.alloc(3)
    assert len(set(a)) == 3 and all(1 <= s <= 7 for s in a)
    assert pool.padding_slot not in a  # slot 0 never allocated
    assert pool.num_free_slots == 4
    pool.free(a)
    assert pool.num_free_slots == 7
    # int and tensor free forms
    s = pool.alloc(1)[0]
    pool.free(s)
    s2 = pool.alloc(2)
    pool.free(torch.tensor(s2, dtype=torch.long))
    assert pool.num_free_slots == 7


def test_alloc_exhaustion_raises():
    pool = _pool(num_slots=4)  # 3 allocatable
    pool.alloc(3)
    with pytest.raises(RuntimeError, match="exhausted"):
        pool.alloc(1)


def test_clear_slots_zeros_all_layers():
    pool = _pool(num_slots=6)
    s = pool.alloc(1)[0]
    pool.conv_states[:, s] = 1.5
    pool.recurrent_states[:, s] = 2.0
    pool.clear_slots([s])
    assert pool.conv_states[:, s].abs().sum() == 0
    assert pool.recurrent_states[:, s].abs().sum() == 0


def test_copy_from_snapshot():
    pool = _pool(num_slots=6)
    src, dst = pool.alloc(2)
    torch.manual_seed(0)
    pool.conv_states[:, src] = torch.randn_like(pool.conv_states[:, src])
    pool.recurrent_states[:, src] = torch.randn_like(pool.recurrent_states[:, src])
    pool.copy_from(src, dst)
    assert torch.equal(pool.conv_states[:, dst], pool.conv_states[:, src])
    assert torch.equal(pool.recurrent_states[:, dst], pool.recurrent_states[:, src])


if __name__ == "__main__":
    test_alloc_free_roundtrip()
    test_alloc_exhaustion_raises()
    test_clear_slots_zeros_all_layers()
    test_copy_from_snapshot()
    print("LinearStatePool allocator unit: PASS")


# qwen4_exp PLE conv-history shape at toy size: one PLE layer, 32 channels, 9 taps
_SPECS = (SlotStateSpec(name="ple_conv", shape=(32, 9), layer_ids=(1,)),)


def test_no_slot_states_by_default():
    pool = _pool()
    assert pool.slot_states == {} and not pool.has_slot_state("ple_conv")
    base = pool.bytes_per_slot()
    pool.clear_slots([1, 2])
    pool.copy_from(1, 2)
    pool.reset(3)
    pool.rebuild(4)
    assert pool.slot_states == {} and pool.bytes_per_slot() == base


def test_slot_state_geometry_and_accessor():
    pool = _pool(num_slots=6, slot_states=_SPECS)
    slab = pool.slot_states["ple_conv"]
    assert slab.shape == (1, 6, 32, 9) and slab.dtype is torch.bfloat16
    assert pool.slot_state("ple_conv", 1).shape == (6, 32, 9)
    with pytest.raises(KeyError):
        pool.slot_state("ple_conv", 0)  # not a declared layer
    with pytest.raises(KeyError):
        pool.slot_state("other")
    with pytest.raises(AssertionError):
        pool.slot_state("ple_conv")  # per-layer state needs layer_id


def test_layerless_spec_dtype_and_fill_value():
    spec = SlotStateSpec(name="ngram", shape=(2,), dtype=torch.int32, fill_value=7.0)
    pool = _pool(num_slots=6, slot_states=(spec,))
    t = pool.slot_state("ngram")
    assert t.shape == (6, 2) and t.dtype is torch.int32 and bool((t == 7).all())
    t[3] = 1
    pool.clear_slots([3])
    assert bool((pool.slot_state("ngram")[3] == 7).all())
    t[2] = 1
    pool.reset(2)
    assert bool((pool.slot_state("ngram")[2] == 7).all())
    pool.rebuild(5)
    assert bool((pool.slot_state("ngram") == 7).all())


def test_duplicate_names_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        _pool(slot_states=_SPECS + _SPECS)


def test_slot_state_follows_every_slot_operation():
    pool = _pool(slot_states=_SPECS)
    pool.slot_states["ple_conv"].fill_(1.0)
    pool.conv_states.fill_(1.0)

    pool.clear_slots([2])
    slab = pool.slot_state("ple_conv", 1)
    assert slab[2].abs().sum().item() == 0.0
    assert slab[3].abs().sum().item() > 0.0

    pool.copy_from(3, 2)
    assert torch.equal(slab[2], slab[3])

    pool.reset(3)
    assert slab[3].abs().sum().item() == 0.0

    pool.rebuild(9)
    slab = pool.slot_state("ple_conv", 1)  # rebuild replaces the tensor
    assert pool.slot_states["ple_conv"].shape == (1, 9, 32, 9)
    assert slab.abs().sum().item() == 0.0

    # the ops write the rebuilt tensor, not a stale alias
    pool.slot_states["ple_conv"].fill_(1.0)
    pool.clear_slots([5])
    assert pool.slot_state("ple_conv", 1)[5].abs().sum().item() == 0.0
    pool.copy_from(1, 5)
    assert torch.equal(pool.slot_state("ple_conv", 1)[5], pool.slot_state("ple_conv", 1)[1])


def test_slot_state_in_the_byte_account():
    group = _group()
    gdn_only = linear_state_bytes_per_req(group, 1, torch.bfloat16)
    with_state = linear_state_bytes_per_req(group, 1, torch.bfloat16, _SPECS)
    assert with_state - gdn_only == 32 * 9 * 2
    assert _pool(slot_states=_SPECS).bytes_per_slot() == with_state

    mc = SimpleNamespace(slot_states=_SPECS)
    mc.linear_attention_group = lambda: group
    config = SimpleNamespace(
        model_config=mc,
        dtype=torch.bfloat16,
        tp_info=SimpleNamespace(size=1),
        cache_type="naive",
        max_running_req=3,
        linear_state_cache_ratio=0.5,
    )
    assert state_pool_bytes(config, num_slots=4) == with_state * 4

    mc.slot_states = ()
    assert state_pool_bytes(config, num_slots=4) == gdn_only * 4

    mc.slot_states = _SPECS
    mc.linear_attention_group = lambda: None
    with pytest.raises(ValueError, match="slot_states"):
        state_pool_bytes(config, num_slots=4)


def test_slot_state_bytes_for_the_real_geometry():
    # qwen4_exp PLE conv history: 4 streams x 2560 channels x 9 taps bf16 = 180 KiB per slot
    spec = SlotStateSpec(name="ple_conv", shape=(4 * 2560, 9), layer_ids=(1,))
    group = _group()
    delta = linear_state_bytes_per_req(
        group, 1, torch.bfloat16, (spec,)
    ) - linear_state_bytes_per_req(group, 1, torch.bfloat16)
    assert delta == 4 * 2560 * 9 * 2 == 180 * 1024


def test_snapshot_cache_ratio_zero_lands_on_the_floor():
    """``--linear-state-cache-ratio 0`` drops the cross-request GDN snapshot cache and sizes
    the pool exactly at the non-evictable floor; positive ratios add ceil(ratio * max_running_req)
    evictable prefix snapshots."""
    from freetoken.kvcache.linear_state_pool import (
        _linear_pool_min_slots,
        _linear_pool_num_slots,
    )

    def cfg(ratio, mr=1):
        return SimpleNamespace(
            max_running_req=mr, cache_type="hybrid_radix", linear_state_cache_ratio=ratio
        )

    assert _linear_pool_num_slots(cfg(2.0)) == 7  # 4 working + 2 snapshots + padding sink
    assert _linear_pool_num_slots(cfg(0.5, mr=3)) == 15  # 12 working + ceil(1.5) + sink
    assert _linear_pool_num_slots(cfg(0.0)) == _linear_pool_min_slots(cfg(0.0)) == 5
    assert _linear_pool_num_slots(cfg(0.0, mr=4)) == _linear_pool_min_slots(cfg(0.0, mr=4)) == 17


@pytest.mark.parametrize("cache_type,working_slots", [("naive", 1), ("hybrid_radix", 4)])
@pytest.mark.parametrize("native_layers", [0, 1])
def test_mtp_rollback_fits_after_non_evictable_slots_are_held(
    cache_type, working_slots, native_layers
):
    from freetoken.kvcache.linear_state_pool import _linear_pool_min_slots, _linear_pool_num_slots

    config = SimpleNamespace(
        max_running_req=1,
        cache_type=cache_type,
        linear_state_cache_ratio=0,
        spec_mtp=1,
        model_config=SimpleNamespace(native_mtp_layers=native_layers),
    )
    slots = _linear_pool_num_slots(config)
    assert slots == _linear_pool_min_slots(config)
    pool = _pool(num_slots=slots)
    pool.alloc(working_slots)
    # Per-row verify buffers cannot restore a partially advanced live slot after OOM.
    assert pool.num_free_slots == 1
    pool.alloc(1)
    assert pool.num_free_slots == 0
