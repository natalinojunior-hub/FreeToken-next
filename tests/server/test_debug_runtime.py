"""CPU-only proof of runtime classification, wire transport, admission and application."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from freetoken.message import (
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    CacheRebuildBackendMsg,
    CacheRebuildMsg,
    CacheRebuildReply,
    CacheRebuildResultMsg,
)
from freetoken.server.runtime import apply_runtime, classify_knob, register_runtime_route


@pytest.fixture(autouse=True)
def _restore_runtime_env():
    """apply_runtime writes os.environ directly; restore it after every test."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("FREETOKEN_MTP_FORCE_DEPTH", "A"),
        ("FREETOKEN_DECODE_RESIDENCY", "A"),
        ("FREETOKEN_VERIFY_GRAPH", "B"),
        ("FREETOKEN_MTP_DRAFT_VOCAB", "B"),
        ("FREETOKEN_SMALL_M_SPLITK", "B"),
        ("FREETOKEN_ROW_INVARIANT_LINEAR", "B"),
        ("FREETOKEN_KV_TIERING", "C"),
        ("FREETOKEN_MTP_BANK", "C"),
        ("max_seq_len", "C"),
        ("PATH", "C"),
        ("FREETOKEN_UNKNOWN", "C"),
    ],
)
def test_classification(name, expected):
    assert classify_knob(name) == expected


def scheduler_stub():
    return SimpleNamespace(spec_mtp=5, engine=SimpleNamespace(moe_offload_cache=None))


def test_apply_and_remove_and_atomic_rejection(monkeypatch):
    monkeypatch.setenv("FREETOKEN_DEBUG_RUNTIME", "1")
    monkeypatch.delenv("FREETOKEN_MTP_FORCE_DEPTH", raising=False)
    scheduler = scheduler_stub()
    result = apply_runtime(scheduler, {"env": {"FREETOKEN_MTP_FORCE_DEPTH": 4}})
    assert result["status"] == "ok"
    assert result["changed"] == {"FREETOKEN_MTP_FORCE_DEPTH": "4"}
    assert os.environ["FREETOKEN_MTP_FORCE_DEPTH"] == "4"
    assert (
        apply_runtime(scheduler, {"env": {"FREETOKEN_MTP_FORCE_DEPTH": 6}})["status"] == "rejected"
    )
    assert (
        apply_runtime(
            scheduler, {"env": {"FREETOKEN_MTP_FORCE_DEPTH": 0, "FREETOKEN_KV_TIERING": "host"}}
        )["status"]
        == "reboot_required"
    )
    assert os.environ["FREETOKEN_MTP_FORCE_DEPTH"] == "4"
    assert apply_runtime(scheduler, {"env": {"FREETOKEN_VERIFY_GRAPH": 0}})["status"] == "rejected"
    apply_runtime(scheduler, {"env": {"FREETOKEN_MTP_FORCE_DEPTH": None}})
    assert "FREETOKEN_MTP_FORCE_DEPTH" not in os.environ
    monkeypatch.delenv("FREETOKEN_DEBUG_RUNTIME")
    assert (
        apply_runtime(scheduler, {"env": {"FREETOKEN_MTP_FORCE_DEPTH": 0}})["status"] == "rejected"
    )


def test_controller_and_stats_reset_preserve_safety(monkeypatch):
    from freetoken.scheduler.adaptive_mtp import AdaptiveMtpController

    monkeypatch.setenv("FREETOKEN_DEBUG_RUNTIME", "1")
    calls = []
    scheduler = scheduler_stub()
    scheduler._mtp_controller = AdaptiveMtpController(5)
    scheduler._mtp_controller.limit_depth(3)
    old = scheduler._mtp_controller
    scheduler._mtp_controllers = {12: old}
    scheduler._mtp_profiled_depth = 5
    scheduler.engine.moe_offload_cache = SimpleNamespace(reset_stats=lambda: calls.append("reset"))
    result = apply_runtime(scheduler, {"reset_mtp_controller": True, "reset_moe_stats": True})
    assert result["status"] == "ok"
    assert scheduler._mtp_controller is not old
    assert scheduler._mtp_controller._runtime_max_k == 3
    assert scheduler._mtp_controllers == {} and scheduler._mtp_profiled_depth is None
    assert calls == ["reset"]


def test_runtime_wire_roundtrip():
    payload = {"env": {"FREETOKEN_MTP_FORCE_DEPTH": None}, "reset_mtp_controller": True}
    cases = [
        (BaseTokenizerMsg, CacheRebuildMsg(request_id="r", runtime=payload)),
        (BaseBackendMsg, CacheRebuildBackendMsg(request_id="r", runtime=payload)),
        (BaseTokenizerMsg, CacheRebuildResultMsg(request_id="r", status="ok", runtime=payload)),
        (BaseFrontendMsg, CacheRebuildReply(request_id="r", status="ok", runtime=payload)),
    ]
    for base, msg in cases:
        assert base.decoder(base.encoder(msg)).runtime == payload


def test_disabled_route_absent(monkeypatch):
    monkeypatch.delenv("FREETOKEN_DEBUG_RUNTIME", raising=False)
    app = FastAPI()
    register_runtime_route(app, None, None)
    with TestClient(app) as client:
        assert client.post("/v1/admin/runtime", json={}).status_code == 404


def test_baked_policy_refresh_and_recapture(monkeypatch):
    import sys

    monkeypatch.setenv("FREETOKEN_DEBUG_RUNTIME", "1")
    monkeypatch.setenv("FREETOKEN_SMALL_M_SPLITK", "1")
    calls = []
    policy = SimpleNamespace(_SMALL_M_DISPATCH=True)
    monkeypatch.setitem(
        sys.modules,
        "freetoken.kernel.triton.fp8_pertensor_linear",
        SimpleNamespace(
            rowwise_scaled_mm_ok=SimpleNamespace(cache_clear=lambda: calls.append("clear"))
        ),
    )
    monkeypatch.setitem(
        sys.modules, "freetoken.layers.quantization.linear", SimpleNamespace(unquantized=policy)
    )
    scheduler = scheduler_stub()
    scheduler.engine.recapture_runtime_graphs = lambda: calls.append("capture")
    result = apply_runtime(
        scheduler, {"env": {"FREETOKEN_SMALL_M_SPLITK": 0}, "recapture_graphs": True}
    )
    assert result["status"] == "ok" and result["recaptured"]
    assert policy._SMALL_M_DISPATCH is False
    assert calls == ["clear", "capture"]


@pytest.mark.parametrize(
    "busy,tp,enabled,status",
    [
        (True, 1, True, "busy"),
        (False, 2, True, "unsupported"),
        (False, 1, False, "rejected"),
        (False, 1, True, None),
    ],
)
def test_scheduler_boundary_gate(monkeypatch, busy, tp, enabled, status):
    from freetoken.scheduler.scheduler import Scheduler

    monkeypatch.setenv("FREETOKEN_DEBUG_RUNTIME", "1" if enabled else "0")
    scheduler = scheduler_stub()
    scheduler.config = SimpleNamespace(tp_info=SimpleNamespace(size=tp))
    scheduler.prefill_manager = SimpleNamespace(runnable=busy)
    scheduler.decode_manager = SimpleNamespace(runnable=False)
    scheduler._pending_rebuild = None
    replies = []
    scheduler._reply_rebuild = lambda request_id, status, *args: replies.append(status)
    msg = CacheRebuildBackendMsg(request_id="r", runtime={"env": {}})
    Scheduler._process_one_msg(scheduler, msg)
    assert replies == ([] if status is None else [status])
    assert scheduler._pending_rebuild is (msg if status is None else None)


def test_scheduler_applies_only_when_pending_boundary_executes(monkeypatch):
    from freetoken.scheduler.scheduler import Scheduler

    monkeypatch.setenv("FREETOKEN_DEBUG_RUNTIME", "1")
    monkeypatch.delenv("FREETOKEN_MTP_FORCE_DEPTH", raising=False)
    scheduler = scheduler_stub()
    scheduler.config = SimpleNamespace(tp_info=SimpleNamespace(size=1))
    scheduler.prefill_manager = SimpleNamespace(runnable=False)
    scheduler.decode_manager = SimpleNamespace(runnable=False)
    scheduler._pending_rebuild = None
    replies = []
    scheduler._reply_rebuild = lambda *args: replies.append(args)
    msg = CacheRebuildBackendMsg(request_id="r", runtime={"env": {"FREETOKEN_MTP_FORCE_DEPTH": 4}})
    Scheduler._process_one_msg(scheduler, msg)
    assert "FREETOKEN_MTP_FORCE_DEPTH" not in os.environ
    Scheduler._execute_pending_rebuild(scheduler)
    assert os.environ["FREETOKEN_MTP_FORCE_DEPTH"] == "4"
    assert scheduler._pending_rebuild is None
    assert replies[0][1] == "ok" and replies[0][3]["applied"] == {"FREETOKEN_MTP_FORCE_DEPTH": "4"}


def test_local_endpoint_validation_and_gate(monkeypatch):
    monkeypatch.setenv("FREETOKEN_DEBUG_RUNTIME", "1")
    monkeypatch.delenv("FREETOKEN_MTP_FORCE_DEPTH", raising=False)
    state = SimpleNamespace(maintenance_state="serving")
    calls = []

    async def dispatch(st, **kwargs):
        calls.append(kwargs)
        assert st is state
        result = apply_runtime(scheduler_stub(), kwargs["runtime"])
        return {"status": result["status"], "runtime": result}

    app = FastAPI()
    register_runtime_route(app, lambda: state, dispatch)
    with TestClient(app, client=("127.0.0.1", 123)) as client:
        response = client.post("/v1/admin/runtime", json={"env": {"FREETOKEN_MTP_FORCE_DEPTH": 0}})
        assert response.status_code == 200
        assert response.json()["applied"] == {"FREETOKEN_MTP_FORCE_DEPTH": "0"}
        assert client.post("/v1/admin/runtime", json={"env": {"PATH": "/tmp"}}).status_code == 409
        assert (
            client.post("/v1/admin/runtime", json={"reset_mtp_controller": "false"}).status_code
            == 422
        )
        assert (
            client.post("/v1/admin/runtime", json={}, headers={"Origin": "http://evil"}).status_code
            == 403
        )
        state.maintenance_state = "rebuilding"
        assert client.post("/v1/admin/runtime", json={}).status_code == 409
    with TestClient(app, client=("192.0.2.1", 123)) as client:
        assert (
            client.post(
                "/v1/admin/runtime", json={}, headers={"X-Forwarded-For": "127.0.0.1"}
            ).status_code
            == 403
        )
    assert len(calls) == 2


def test_frontend_dispatch_blocks_and_replies_without_polling():
    from freetoken.server.api_server import FrontendManager, dispatch_rebuild

    async def check():
        state = FrontendManager(config=SimpleNamespace(), send_tokenizer=None, recv_tokenizer=None)
        state.maintenance_state = "serving"

        async def send(msg):
            assert state.maintenance_state == "rebuilding"
            state._resolve_rebuild(
                CacheRebuildReply(
                    request_id=msg.request_id, status="ok", runtime={"applied": msg.runtime["env"]}
                )
            )

        state.send_one = send
        result = await dispatch_rebuild(
            state, moe_cache_size=None, num_pages=None, runtime={"env": {}}
        )
        assert result["runtime"] == {"applied": {}}
        assert state.maintenance_state == "serving" and not state.rebuild_futures

    asyncio.run(check())
