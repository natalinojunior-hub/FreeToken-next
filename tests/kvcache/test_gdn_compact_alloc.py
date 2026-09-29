"""CPU accounting checks for compact speculative GDN state buffers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.kvcache.linear_state_pool import (
    LinearStatePool,
    spec_state_bytes,
    ssm_state_dtype,
)
from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec


def _config(steps: int):
    group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=(0, 1),
        num_key_heads=2,
        num_value_heads=4,
        key_head_dim=4,
        value_head_dim=5,
        conv_kernel_dim=3,
        output_gate="silu",
    )
    siblings = (
        SlotStateSpec("sibling_per_layer", (2, 3), layer_ids=(0, 1)),
        SlotStateSpec("sibling_global", (5,), dtype=torch.int32),
    )
    model = SimpleNamespace(
        mtp_row_state_commit=True,
        native_mtp_layers=0,
        slot_states=siblings,
        linear_attention_group=lambda: group,
    )
    config = SimpleNamespace(
        spec_mtp=steps - 1,
        model_config=model,
        tp_info=SimpleNamespace(size=1),
        dtype=torch.bfloat16,
    )
    return config, group, siblings


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


@pytest.mark.parametrize("steps", [3, 5])
def test_compact_spec_buffer_allocation_matches_byte_estimate(monkeypatch, steps):
    monkeypatch.setenv("FREETOKEN_MTP_COMPACT_STATE", "1")
    monkeypatch.setenv("FREETOKEN_MTP_ROW_COMMIT", "1")
    config, group, siblings = _config(steps)
    pool = LinearStatePool(
        group,
        num_slots=3,
        dtype=config.dtype,
        device=torch.device("cpu"),
        tp_size=1,
        slot_states=siblings,
        spec_steps=steps,
    )

    actual = sum(
        _nbytes(t)
        for t in (
            pool.spec_states,
            pool.spec_conv_in,
            pool.spec_conv_pre,
            pool.spec_qkv,
            pool.spec_ba,
            *pool.spec_slot_states.values(),
        )
    )
    assert actual == spec_state_bytes(config)

    layers = len(group.layer_ids)
    conv_dim = 2 * group.num_key_heads * group.key_head_dim + (
        group.num_value_heads * group.value_head_dim
    )
    recurrent_bytes = (
        group.num_value_heads
        * group.key_head_dim
        * group.value_head_dim
        * ssm_state_dtype().itemsize
    )
    conv_bytes = conv_dim * config.dtype.itemsize
    sibling_bytes_per_row = sum(
        max(1, len(spec.layer_ids))
        * torch.empty((), dtype=spec.dtype or config.dtype).element_size()
        * torch.tensor(spec.shape).prod().item()
        for spec in siblings
    )
    full_checkpoint_bytes = (
        layers * (steps * (recurrent_bytes + conv_bytes) + conv_bytes * (group.conv_kernel_dim - 1))
        + steps * sibling_bytes_per_row
    )
    assert actual < full_checkpoint_bytes
