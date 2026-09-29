"""Prompt draft KV pairs target residuals with successor tokens, across chunks."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from freetoken.core import Batch, Req, SamplingParams
from freetoken.scheduler.spec import SchedulerSpecMixin


def _scheduler():
    calls = []

    def forward(residual, tokens, batch):
        assert batch.reqs[0].cached_len == int(batch.positions[0])
        assert batch.reqs[0].device_len == int(batch.positions[-1]) + 1
        assert batch.spec_logits_indices is None
        calls.append(
            (residual.clone(), tokens.clone(), batch.positions.clone(), batch.out_loc.clone())
        )

    scheduler = SchedulerSpecMixin()
    scheduler.spec_mtp = 1
    scheduler.device = torch.device("cpu")
    scheduler._model_is_mrope = False
    scheduler._spec_snapshot_slots = {}
    scheduler.engine = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(), mtp=SimpleNamespace(forward=forward)),
        page_table=torch.arange(32, dtype=torch.int32).reshape(1, -1),
        attn_backend=SimpleNamespace(prepare_metadata=lambda batch: None),
        ctx=SimpleNamespace(forward_batch=lambda batch: nullcontext()),
    )
    return scheduler, calls


def _target(scheduler, start, end, uid=7):
    req = Req(
        input_ids=torch.arange(100, 100 + end, dtype=torch.int32),
        table_idx=0,
        cached_len=start,
        output_len=4,
        uid=uid,
        sampling_params=SamplingParams(),
        cache_handle=SimpleNamespace(cached_len=start),
    )
    batch = Batch(reqs=[req], phase="prefill")
    batch.input_ids = req.input_ids[start:end].clone()
    batch.positions = torch.arange(start, end, dtype=torch.int32)
    scheduler.engine.model.model._last_residual = torch.arange(start, end).float().reshape(-1, 1)
    req.complete_one()
    req.append_host(torch.tensor([999], dtype=torch.int32))  # final sample must stay unwarmed
    return req, batch


def _assert_rows(call, residuals, tokens, positions):
    residual, actual_tokens, actual_positions, out_loc = call
    assert residual[:, 0].tolist() == residuals
    assert actual_tokens.tolist() == tokens
    assert actual_positions.tolist() == positions
    assert out_loc.tolist() == positions


def test_cold_and_contiguous_chunks_pair_successors_without_final_sample():
    scheduler, calls = _scheduler()
    req, batch = _target(scheduler, 0, 4)
    scheduler.warmup_mtp_draft_kv(req, batch)
    _assert_rows(calls[0], [0, 1, 2], [101, 102, 103], [1, 2, 3])
    assert (req.cached_len, req.device_len) == (4, 5)

    req, batch = _target(scheduler, 4, 7)
    scheduler.warmup_mtp_draft_kv(req, batch)
    _assert_rows(calls[1], [3, 4, 5], [104, 105, 106], [4, 5, 6])
    assert (req.cached_len, req.device_len) == (7, 8)
    assert scheduler._mtp_prompt_carry[:2] == (7, 7)
    assert scheduler._mtp_prompt_carry[2].tolist() == [[6]]


@pytest.mark.parametrize("carry", [None, (7, 3), (8, 4)])
def test_radix_suffix_and_unrelated_carry_do_not_supply_missing_residual(carry):
    scheduler, calls = _scheduler()
    if carry is not None:
        scheduler._mtp_prompt_carry = (*carry, torch.tensor([[-999.0]]))
    req, batch = _target(scheduler, 4, 7)
    scheduler.warmup_mtp_draft_kv(req, batch)
    _assert_rows(calls[0], [4, 5], [105, 106], [5, 6])


def test_one_token_chunk_keeps_carry_for_next_chunk():
    scheduler, calls = _scheduler()
    req, batch = _target(scheduler, 0, 1)
    scheduler.warmup_mtp_draft_kv(req, batch)
    assert calls == []
    req, batch = _target(scheduler, 1, 2)
    scheduler.warmup_mtp_draft_kv(req, batch)
    _assert_rows(calls[0], [0], [101], [1])


def test_forward_exception_restores_request_lengths():
    scheduler, _ = _scheduler()
    req, batch = _target(scheduler, 0, 4)

    def fail(*args):
        raise RuntimeError("draft failed")

    scheduler.engine.model.mtp.forward = fail
    with pytest.raises(RuntimeError, match="draft failed"):
        scheduler.warmup_mtp_draft_kv(req, batch)
    assert (req.cached_len, req.device_len) == (4, 5)


def test_cleanup_discards_only_matching_prompt_carry():
    scheduler, _ = _scheduler()
    req, batch = _target(scheduler, 0, 4)
    scheduler.warmup_mtp_draft_kv(req, batch)
    scheduler.free_spec_snapshot_slot(SimpleNamespace(uid=8))
    assert scheduler._mtp_prompt_carry[:2] == (7, 4)
    scheduler.free_spec_snapshot_slot(req)
    assert scheduler._mtp_prompt_carry is None
