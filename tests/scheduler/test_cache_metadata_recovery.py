"""Metadata OOM retries require actual backing release before ownership transfer."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.scheduler.scheduler import Scheduler


@pytest.mark.parametrize("after,expected", [(8, True), (10, False), (12, False)])
def test_cache_metadata_recovery_notes_before_shrink_and_requires_freed_rows(
    monkeypatch, after, expected
):
    events = []
    cache = SimpleNamespace(resident_rows=10, _vmm_arenas=[object()])
    error = torch.OutOfMemoryError("Tried to allocate 2.00 MiB")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append(("sync", device)))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty"))

    def shrink():
        events.append("shrink")
        cache.resident_rows = after

    scheduler = SimpleNamespace(
        device="cpu",
        engine=SimpleNamespace(
            moe_offload_cache=cache,
            note_decode_oom=lambda exc: events.append(("note", exc)),
            shrink_after_oom=shrink,
        ),
    )
    assert Scheduler._recover_cache_oom(scheduler, error) is expected
    assert events == [("note", error), ("sync", "cpu"), "shrink", "empty"]


@pytest.mark.parametrize("missing", ["cache", "shrink", "vmm"])
def test_cache_metadata_recovery_without_release_mechanism_is_not_a_retry(monkeypatch, missing):
    def unexpected(*_args):
        pytest.fail("recovery must not start without backing release")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected)
    monkeypatch.setattr(torch.cuda, "empty_cache", unexpected)
    engine = SimpleNamespace(note_decode_oom=unexpected)
    if missing != "cache":
        engine.moe_offload_cache = SimpleNamespace(
            resident_rows=10, _vmm_arenas=[] if missing == "vmm" else [object()]
        )
    if missing != "shrink":
        engine.shrink_after_oom = unexpected
    assert (
        Scheduler._recover_cache_oom(
            SimpleNamespace(engine=engine, device="cpu"), torch.OutOfMemoryError("out of memory")
        )
        is False
    )
