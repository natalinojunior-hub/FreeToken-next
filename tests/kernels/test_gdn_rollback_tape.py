"""Exactness and graph-safety checks for the fused GDN rollback tape."""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEVICE = torch.device("cuda")
HEADS, VALUE_HEADS, KEY_DIM, VALUE_DIM = 3, 6, 128, 128


def _inputs(steps: int, seed: int):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    q = torch.randn(1, steps, HEADS, KEY_DIM, generator=gen, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(
        1, steps, VALUE_HEADS, VALUE_DIM, generator=gen, device=DEVICE, dtype=torch.bfloat16
    )
    a = torch.randn(steps, VALUE_HEADS, generator=gen, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    A_log = torch.randn(VALUE_HEADS, generator=gen, device=DEVICE, dtype=torch.float32)
    dt_bias = torch.randn(VALUE_HEADS, generator=gen, device=DEVICE, dtype=torch.float32)
    initial = torch.randn(
        1, VALUE_HEADS, KEY_DIM, VALUE_DIM, generator=gen, device=DEVICE, dtype=torch.float32
    )
    return q, k, v, a, b, A_log, dt_bias, initial


def _run(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    state,
    *,
    tape=None,
    indices=None,
    intermediate=None,
    gate_batch_stride=0,
):
    batch, steps = q.shape[:2]
    if indices is None:
        indices = torch.zeros(batch, dtype=torch.int32, device=DEVICE)
    cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device=DEVICE) * steps
    return fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=state,
        initial_state_indices=indices,
        scale=KEY_DIM**-0.5,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
        intermediate_states_buffer=intermediate,
        intermediate_state_indices=indices if intermediate is not None else None,
        gate_batch_stride=gate_batch_stride,
        rollback_tape=tape,
    )


@pytest.mark.parametrize("steps", (2, 3, 5))
def test_rollback_tape_matches_normal_and_is_cuda_graph_safe(steps):
    q, k, v, a, b, A_log, dt_bias, initial = _inputs(steps, 31 + steps)
    indices = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    row_width = 2 * HEADS * KEY_DIM + VALUE_HEADS * VALUE_DIM

    normal_state = initial.clone()
    normal_rows = torch.empty(
        1, steps, VALUE_HEADS, KEY_DIM, VALUE_DIM, device=DEVICE, dtype=torch.float32
    )
    normal = _run(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        normal_state,
        indices=indices,
        intermediate=normal_rows,
    )

    saved_initial = torch.empty_like(initial[0])
    saved_qkv = torch.empty(steps, row_width, device=DEVICE, dtype=torch.bfloat16)
    saved_ba = torch.empty(steps, 2 * VALUE_HEADS, device=DEVICE, dtype=torch.bfloat16)
    tape_state = initial.clone()
    taped = _run(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        tape_state,
        tape=(saved_initial, saved_qkv, saved_ba),
        indices=indices,
    )
    assert torch.equal(taped, normal)
    assert torch.equal(tape_state, normal_state)
    assert torch.equal(saved_initial, initial[0])
    expected_qkv = torch.cat(
        [q[0].reshape(steps, -1), k[0].reshape(steps, -1), v[0].reshape(steps, -1)], dim=-1
    )
    expected_ba = torch.cat([b, a], dim=-1)
    assert torch.equal(saved_qkv, expected_qkv)
    assert torch.equal(saved_ba, expected_ba)

    graph_state = initial.clone()
    graph_initial = torch.empty_like(saved_initial)
    graph_qkv = torch.empty_like(saved_qkv)
    graph_ba = torch.empty_like(saved_ba)
    tape = (graph_initial, graph_qkv, graph_ba)
    _run(q, k, v, a, b, A_log, dt_bias, graph_state, tape=tape, indices=indices)
    torch.cuda.synchronize()
    graph_state.copy_(initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run(q, k, v, a, b, A_log, dt_bias, graph_state, tape=tape, indices=indices)

    graph_state.copy_(initial)
    graph_initial.zero_()
    graph_qkv.zero_()
    graph_ba.zero_()
    graph.replay()
    assert torch.equal(captured, normal)
    assert torch.equal(graph_state, normal_state)
    assert torch.equal(graph_initial, initial[0])
    assert torch.equal(graph_qkv, expected_qkv)
    assert torch.equal(graph_ba, expected_ba)


@pytest.mark.parametrize("state_dtype", (torch.float32, torch.bfloat16))
def test_two_row_recurrence_matches_successive_single_rows(state_dtype):
    """MTP verify must leave the same continuation state as RAW decode."""
    q, k, v, a, b, A_log, dt_bias, initial = _inputs(2, 811)
    initial = initial.to(state_dtype)
    batched_state = initial.clone()
    batched_rows = torch.empty(
        1, 2, VALUE_HEADS, KEY_DIM, VALUE_DIM, device=DEVICE, dtype=torch.float32
    )
    batched_out = _run(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        batched_state,
        intermediate=batched_rows,
    )

    sequential_state = initial.clone()
    sequential_out = []
    sequential_rows = []
    for row in range(2):
        checkpoints = torch.empty(
            1, 1, VALUE_HEADS, KEY_DIM, VALUE_DIM, device=DEVICE, dtype=torch.float32
        )
        output = _run(
            q[:, row : row + 1],
            k[:, row : row + 1],
            v[:, row : row + 1],
            a[row : row + 1],
            b[row : row + 1],
            A_log,
            dt_bias,
            sequential_state,
            intermediate=checkpoints,
        )
        sequential_out.append(output)
        sequential_rows.append(checkpoints[:, 0])

    assert torch.equal(batched_out, torch.cat(sequential_out, dim=1))
    assert torch.equal(batched_rows, torch.stack(sequential_rows, dim=1)[0:1])
    assert torch.equal(batched_state, sequential_state)


@pytest.mark.parametrize("steps", (2, 3, 5))
def test_batched_layer_prefix_replay_matches_row_checkpoints(steps):
    """Replay every target-layer prefix from its taped S0 in one batch. Each batch item has
    layer-specific A_log/dt_bias via gate_batch_stride; tape rows retain the exact BF16 inputs."""
    layer_inputs = [_inputs(steps, 101 * layer + steps) for layer in range(3)]
    row_width = 2 * HEADS * KEY_DIM + VALUE_HEADS * VALUE_DIM
    initial_rows = []
    qkv_rows = []
    ba_rows = []
    checkpoints = []

    for q, k, v, a, b, A_log, dt_bias, initial in layer_inputs:
        reference_state = initial.clone()
        rows = torch.empty(
            1, steps, VALUE_HEADS, KEY_DIM, VALUE_DIM, device=DEVICE, dtype=torch.float32
        )
        _run(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            reference_state,
            intermediate=rows,
        )

        tape_state = initial.clone()
        initial_tape = torch.empty_like(initial[0])
        qkv_tape = torch.empty(steps, row_width, device=DEVICE, dtype=torch.bfloat16)
        ba_tape = torch.empty(steps, 2 * VALUE_HEADS, device=DEVICE, dtype=torch.bfloat16)
        _run(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            tape_state,
            tape=(initial_tape, qkv_tape, ba_tape),
        )
        assert torch.equal(initial_tape, initial[0])
        initial_rows.append(initial_tape)
        qkv_rows.append(qkv_tape)
        ba_rows.append(ba_tape)
        checkpoints.append(rows[0])

    initial_state = torch.stack(initial_rows)
    qkv = torch.stack(qkv_rows)
    ba = torch.stack(ba_rows)
    q = qkv[..., : HEADS * KEY_DIM].reshape(3, steps, HEADS, KEY_DIM).contiguous()
    k_lo, k_hi = HEADS * KEY_DIM, 2 * HEADS * KEY_DIM
    k = qkv[..., k_lo:k_hi].reshape(3, steps, HEADS, KEY_DIM).contiguous()
    v = qkv[..., k_hi:].reshape(3, steps, VALUE_HEADS, VALUE_DIM).contiguous()
    b = ba[..., :VALUE_HEADS].contiguous()
    a = ba[..., VALUE_HEADS:].contiguous()
    A_log = torch.stack([item[5] for item in layer_inputs]).contiguous()
    dt_bias = torch.stack([item[6] for item in layer_inputs]).contiguous()
    indices = torch.arange(3, dtype=torch.int32, device=DEVICE)
    expected = torch.stack(checkpoints)

    for prefix in range(1, steps + 1):
        replay_state = initial_state.clone()
        replay = _run(
            q[:, :prefix].contiguous(),
            k[:, :prefix].contiguous(),
            v[:, :prefix].contiguous(),
            a[:, :prefix].contiguous(),
            b[:, :prefix].contiguous(),
            A_log,
            dt_bias,
            replay_state,
            indices=indices,
            gate_batch_stride=VALUE_HEADS,
        )
        assert replay.shape[0] == 3
        assert torch.equal(replay_state, expected[:, prefix - 1]), (
            f"batched state mismatch after prefix {prefix}"
        )
