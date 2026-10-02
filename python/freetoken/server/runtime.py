"""Local debug runtime controls; no generation-path work when debug mode is off.

Knob classification (unlisted knobs conservatively require a reboot):

Class | Knobs                              | Boundary action
A     | MTP_FORCE_DEPTH, DECODE_RESIDENCY,  | Update Python's per-cycle/request env.
      | DEBUG_MTP_CYCLES, DEBUG_SPEC_TIMING,|
      | SPEC_LOOKUP, SPEC_DEFER_REPLAY,     |
      | ENABLE_PARTIAL_SPEC               |
B     | VERIFY_GRAPH, DRAFT_GRAPH,         | Destroy/re-capture existing graph sizes;
      | MTP_DRAFT_VOCAB, SMALL_M_SPLITK,    | refresh module dispatch policy and clear
      | ROW_INVARIANT_LINEAR              | FP8 dispatch capability cache first.
C     | KV tier/dtype, expert-bank layout, | Reboot required; no env mutation.
      | MTP bank/spec_mtp, max_seq_len,    |
      | all other/unknown knobs           |

Names above have the FREETOKEN_ prefix. Debug mode is opt-in via
FREETOKEN_DEBUG_RUNTIME=1 at boot. Only the actual loopback peer is trusted;
forwarded headers are ignored and browser Origin requests are rejected.
TP > 1 is unsupported, matching the existing single-rank rebuild protocol.
"""

from __future__ import annotations

import ipaddress
import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

IMMEDIATE = frozenset(
    "FREETOKEN_" + name
    for name in (
        "MTP_FORCE_DEPTH",
        "DECODE_RESIDENCY",
        "DEBUG_MTP_CYCLES",
        "DEBUG_SPEC_TIMING",
        "SPEC_LOOKUP",
        "SPEC_DEFER_REPLAY",
        "ENABLE_PARTIAL_SPEC",
    )
)
RECAPTURE = frozenset(
    "FREETOKEN_" + name
    for name in (
        "VERIFY_GRAPH",
        "DRAFT_GRAPH",
        "MTP_DRAFT_VOCAB",
        "SMALL_M_SPLITK",
        "ROW_INVARIANT_LINEAR",
    )
)


def classify_knob(name: str) -> Literal["A", "B", "C"]:
    return "A" if name in IMMEDIATE else "B" if name in RECAPTURE else "C"


class RuntimeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    env: dict[str, StrictStr | StrictInt | None] = Field(default_factory=dict)
    recapture_graphs: StrictBool = False
    reset_mtp_controller: StrictBool = False
    reset_moe_stats: StrictBool = False


def apply_runtime(scheduler: Any, payload: dict) -> dict:
    """Called ONLY at the scheduler's existing fully-drained rebuild boundary."""
    request = RuntimeRequest.model_validate(payload)
    if os.environ.get("FREETOKEN_DEBUG_RUNTIME") != "1":
        return {"status": "rejected", "error": "debug runtime is disabled"}
    values = {k: None if v is None else str(v) for k, v in request.env.items()}
    classes = {k: classify_knob(k) for k in values}
    reboot = [k for k in values if classes[k] == "C"]
    if reboot:
        return {"status": "reboot_required", "reboot_required": reboot, "classification": classes}
    changed = {k: v for k, v in values.items() if os.environ.get(k) != v}
    baked = [k for k in changed if classes[k] == "B"]
    if baked and not request.recapture_graphs:
        return {
            "status": "rejected",
            "error": "recapture_graphs required",
            "recapture_required": baked,
        }
    for k, v in values.items():
        if v is None:
            continue
        if k == "FREETOKEN_MTP_FORCE_DEPTH":
            if not v.isdecimal() or not 0 <= int(v) <= scheduler.spec_mtp:
                return {
                    "status": "rejected",
                    "error": f"force depth must be 0..{scheduler.spec_mtp}",
                }
        elif k == "FREETOKEN_MTP_DRAFT_VOCAB":
            if not v.isdecimal() or int(v) < 1:
                return {"status": "rejected", "error": "draft vocab must be a positive integer"}
        elif v not in {"0", "1"}:
            return {"status": "rejected", "error": f"{k} must be 0 or 1"}
    for k, v in changed.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    if request.recapture_graphs:
        from freetoken.kernel.triton.fp8_pertensor_linear import rowwise_scaled_mm_ok
        from freetoken.layers.quantization.linear import unquantized

        unquantized._SMALL_M_DISPATCH = os.environ.get("FREETOKEN_SMALL_M_SPLITK", "1") != "0"
        rowwise_scaled_mm_ok.cache_clear()
        scheduler.engine.recapture_runtime_graphs()
    if request.reset_mtp_controller:
        from freetoken.scheduler.adaptive_mtp import AdaptiveMtpController

        base = getattr(scheduler, "_mtp_controller", None)
        if base is not None:
            # Retain OOM safety limits, discard the arm's learned economics/profile.
            scheduler._mtp_controller = AdaptiveMtpController(base.safe_max_k)
            scheduler._mtp_controller.limit_depth(base._runtime_max_k)
        scheduler._mtp_controllers = {}
        scheduler._mtp_profiled_depth = None
    if request.reset_moe_stats and scheduler.engine.moe_offload_cache is not None:
        scheduler.engine.moe_offload_cache.reset_stats()
    return {
        "status": "ok",
        "applied": values,
        "changed": changed,
        "classification": classes,
        "recaptured": request.recapture_graphs,
        "reset_mtp_controller": request.reset_mtp_controller,
        "reset_moe_stats": request.reset_moe_stats,
    }


def register_runtime_route(app: FastAPI, get_state: Any, dispatch: Any) -> None:
    if os.environ.get("FREETOKEN_DEBUG_RUNTIME") != "1":
        return

    @app.post("/v1/admin/runtime")
    async def runtime(request: Request, body: RuntimeRequest):
        peer = request.client.host if request.client else ""
        try:
            local = ipaddress.ip_address(peer).is_loopback
        except ValueError:
            local = False
        if not local or request.headers.get("origin"):
            raise HTTPException(403, "loopback operator requests only")
        state = get_state()
        if state.maintenance_state != "serving":
            raise HTTPException(409, "scheduler maintenance already in progress")
        if getattr(state, "ack_map", None):
            raise HTTPException(409, "generation requests are still admitted")
        result = await dispatch(
            state, moe_cache_size=None, num_pages=None, runtime=body.model_dump()
        )
        status = result.get("status")
        if status == "ok":
            return result["runtime"]
        raise HTTPException(
            409 if status in {"busy", "rejected", "unsupported", "reboot_required"} else 503, result
        )
