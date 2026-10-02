"""Batched speculative failures leave every stream at its unpublished baseline."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.spec import SchedulerSpecMixin


@pytest.mark.parametrize("failure", ["draft", "verify", "replay", None])
@pytest.mark.parametrize("second_depth", [0, 1])
def test_batched_spec_transaction(monkeypatch, failure, second_depth):
    monkeypatch.setenv("FREETOKEN_MTP_BATCHED", "1")
    reqs = [
        Req(
            input_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
            cached_len=2,
            table_idx=i,
            uid=i,
            output_len=3,
            cache_handle=None,
            sampling_params=SamplingParams(max_tokens=8, temperature=0),
        )
        for i in range(2)
    ]
    for req in reqs:
        req.device_len = 3
        req.alloc_page_bound = 3
    states = torch.arange(16, dtype=torch.float32).view(4, 4)
    before = states[:2].clone()
    carry = {i: (2, torch.ones(1, 1), torch.tensor([3])) for i in range(2)}
    residuals = {i: torch.tensor([[float(i)]]) for i in range(2)}
    published, freed = [], []

    def free_pages(req, keep_len, alloc_len):
        req.alloc_page_bound = keep_len

    def copy(src, dst):
        states[dst].copy_(states[src])

    scheduler = SimpleNamespace(
        spec_mtp=1,
        device=torch.device("cpu"),
        _mtp_controllers={
            i: SimpleNamespace(next_depth=lambda depth=depth: depth)
            for i, depth in enumerate([1, second_depth])
        },
        _mtp_kv_rows_map=carry.copy(),
        _mtp_residual_by_uid=residuals.copy(),
        _spec_cycle_reqs=reqs,
        _mtp_kv_rows=None,
        token_pool=torch.ones((2, 16), dtype=torch.int64),
        finished_reqs=set(),
        eos_token_ids={99},
        toolcall_anchor_id=None,
        cache_manager=SimpleNamespace(
            page_size=1,
            lazy_free_region=nullcontext,
            free_spec_reject=free_pages,
            cache_req=lambda *a, **kw: None,
        ),
        decode_manager=SimpleNamespace(filter_reqs=lambda _: None, remove_req=lambda _: None),
        _free_req_resources=freed.append,
        send_result=published.extend,
        _flush_deferred_replays=lambda: None,
        _snapshot_qsa_state=lambda _: None,
        _snapshot_ple_state=lambda _: None,
        _restore_qsa_state=lambda *a, **kw: None,
        _restore_ple_state=lambda _: None,
        _retain_mtp_ring=lambda _: None,
        _spec_snapshot_slot=lambda req: req.table_idx + 2,
        _linear_slot=lambda req: req.table_idx,
        _match_stop_str=lambda _: None,
    )
    model_state = SimpleNamespace(_last_residual=torch.zeros((1, 1)))
    scheduler.engine = SimpleNamespace(
        linear_state_pool=SimpleNamespace(copy_from=copy),
        model=SimpleNamespace(model=model_state),
    )
    scheduler._batched_maps = lambda: SchedulerSpecMixin._batched_maps(scheduler)
    scheduler._take_mtp_fill = lambda req: SchedulerSpecMixin._take_mtp_fill(scheduler, req)
    scheduler._commit_spec_tokens = lambda *a, **kw: SchedulerSpecMixin._commit_spec_tokens(
        scheduler, *a, **kw
    )

    def draft(req, pos, residual, tokens):
        states[req.table_idx].add_(20)
        req.alloc_page_bound = 4
        if failure == "draft" and req.uid == (1 if second_depth else 0):
            raise torch.OutOfMemoryError("draft injection")
        return residual[-1:], torch.zeros((1, 1)), torch.tensor([10])

    def prepare(batch):
        rows = 3 + second_depth
        for req, depth in zip(reqs, [1, second_depth]):
            req.alloc_page_bound = 3 + depth
        return SimpleNamespace(
            input_tuple=(torch.zeros(rows, dtype=torch.long), torch.arange(rows)),
            sample_args=None,
        )

    def forward(batch, sample_args):
        states[:2].add_(30)
        model_state._last_residual = torch.arange(3 + second_depth).view(-1, 1).float()
        if failure == "verify":
            raise torch.OutOfMemoryError("verify injection")
        sampled = torch.tensor([99, 20, 88] + ([21] if second_depth else []))
        return SimpleNamespace(
            next_tokens_cpu=sampled,
            next_tokens_gpu=sampled,
            copy_done_event=SimpleNamespace(synchronize=lambda: None),
        )

    def replay(req, start, count):
        states[req.table_idx].add_(40)
        if failure == "replay" and req.uid == 1:
            raise torch.OutOfMemoryError("later replay injection")

    scheduler._draft_step = draft
    scheduler._prepare_batch = prepare
    scheduler.engine.forward_batch = forward
    scheduler._replay = replay
    will_fail = failure in {"draft", "verify"} or (failure == "replay" and second_depth)
    if will_fail:
        with pytest.raises(torch.OutOfMemoryError):
            SchedulerSpecMixin._batched_spec_cycle(scheduler)
        assert published == [] and freed == []
        assert scheduler._spec_rollback is not None
        scheduler._spec_rollback()
        assert torch.equal(states[:2], before)
        assert all(req.input_ids.tolist() == [1, 2, 3] for req in reqs)
        assert all((req.cached_len, req.device_len) == (2, 3) for req in reqs)
        assert all(req.alloc_page_bound == 3 for req in reqs)
        assert scheduler._mtp_kv_rows_map == carry
        assert scheduler._mtp_residual_by_uid == residuals
    else:
        assert SchedulerSpecMixin._batched_spec_cycle(scheduler)
        assert scheduler._spec_rollback is None
        assert [msg.uid for msg in published] == [0, 1]
        assert freed == [reqs[0]]
        assert reqs[0] in scheduler.finished_reqs
