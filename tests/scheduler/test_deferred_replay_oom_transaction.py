"""Protected replay retries every owed target row before allowing plain decode."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.spec import SchedulerSpecMixin


@pytest.mark.parametrize("failure", ["snapshot", "replay", None])
@pytest.mark.parametrize("failed_uid", [0, 1])
def test_deferred_replay_is_transactional_per_request(monkeypatch, failure, failed_uid):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    reqs = [
        SimpleNamespace(
            uid=i,
            table_idx=i,
            cached_len=2,
            device_len=5,
            input_ids=torch.tensor([1, 2, 3, 4, 5]),
        )
        for i in range(2)
    ]
    linear = torch.arange(16).view(4, 4).float()
    qsa = torch.arange(8).view(2, 4).float()
    before_linear, before_qsa = linear[:2].clone(), qsa.clone()
    preimages, active = {}, {}

    def snapshot_qsa(table_idx, *args, **kwargs):
        preimages[table_idx] = qsa[table_idx].clone()

    def abort_qsa(table_idx):
        saved = active.pop(table_idx, preimages.pop(table_idx, None))
        if saved is not None:
            qsa[table_idx].copy_(saved)

    def prepare_qsa(table_idx, *args, **kwargs):
        active[table_idx] = preimages.pop(table_idx)

    def commit_qsa(table_idx):
        active.pop(table_idx, None)
        preimages.pop(table_idx, None)

    backend = SimpleNamespace(
        snapshot_spec_preimage=snapshot_qsa,
        prepare_spec_txn=prepare_qsa,
        abort_spec_txn=abort_qsa,
        commit_spec_txn=commit_qsa,
        _spec_preimages=preimages,
        _restore_txn=lambda image: None,
    )
    model_state = SimpleNamespace(_last_residual=torch.zeros((1, 1)))
    scheduler = SimpleNamespace(
        device=torch.device("cpu"),
        decode_manager=SimpleNamespace(running_reqs=reqs),
        _snapshot_qsa_state=lambda req: None,
        _restore_qsa_state=lambda req, **kwargs: None,
        _linear_slot=lambda req: req.table_idx,
        engine=SimpleNamespace(
            model=SimpleNamespace(model=model_state),
            attn_backend=backend,
            page_table=torch.arange(16).view(2, 8),
            linear_state_pool=SimpleNamespace(
                copy_from=lambda src, dst: linear[dst].copy_(linear[src]),
            ),
            shrink_after_oom=lambda: None,
        ),
    )
    replayed = []

    def snapshot_slot(req):
        if failure == "snapshot" and req.uid == failed_uid:
            raise torch.OutOfMemoryError("snapshot injection")
        return req.table_idx + 2

    def replay(req, start, count):
        replayed.append((req.uid, start, count))
        linear[req.table_idx].add_(20)
        qsa[req.table_idx].add_(30)
        model_state._last_residual = torch.tensor([[float(req.uid + 40)]])
        req.cached_len, req.device_len = 4, 5
        if failure == "replay" and req.uid == failed_uid:
            raise torch.OutOfMemoryError("replay injection")

    scheduler._spec_snapshot_slot = snapshot_slot
    scheduler._replay = replay
    scheduler.run_spec_step = lambda: SchedulerSpecMixin._flush_deferred_replays(scheduler) or True
    assert Scheduler._spec_step_or_fail(scheduler) is True
    assert not active and not preimages
    if failure is not None:
        assert scheduler._mtp_cycle_observe is False
        assert scheduler._spec_rollback is None
        assert (reqs[failed_uid].cached_len, reqs[failed_uid].device_len) == (2, 5)
        assert torch.equal(linear[failed_uid], before_linear[failed_uid])
        assert torch.equal(qsa[failed_uid], before_qsa[failed_uid])
        if failed_uid:
            assert reqs[0].cached_len == 4  # prior request finalized independently
            assert torch.equal(linear[0], before_linear[0] + 20)
    else:
        assert replayed == [(0, 2, 2), (1, 2, 2)]
        assert all((req.cached_len, req.device_len) == (4, 5) for req in reqs)
        assert torch.equal(linear[:2], before_linear + 20)
        assert torch.equal(qsa, before_qsa + 30)
