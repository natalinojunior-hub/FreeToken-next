"""Persistent engine diagnostics: OOM, auto-degradations, and other problems appended to a
fingerprinted JSONL so the operator can investigate later and harden the auto system.

The engine is 100% automatic (it sheds MTP, steps KV format down, falls back from VMM-lazy
expert residency, halves the calibration chunk on OOM, ...). Those decisions are the auto
system trading performance for safety, and today they only print to the server log, which is
lost when the process exits. This module persists them, keyed on the same hardware + model +
build fingerprint as ``tuning.profile`` / ``tuning.history``, one JSON object per line:

    {"ts": "...", "kind": "degrade", "phase": "shed_mtp", "severity": "warn",
     "message": "--spec-mtp 4 -> 0 ...", "context": {...config/vram snapshot...}}

Best-effort by construction: ``log_event`` NEVER raises (a diagnostics write must not take down
serving), and an unconfigured module silently drops events. Configure once at engine boot via
``configure(model_path, config_summary)``; every later ``log_event`` is attributed to that
fingerprint. ``FREETOKEN_DIAGNOSTICS=0`` disables persistence entirely (events still go to the
normal logger).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from freetoken.moe.bench_profile import _cache_dir
from freetoken.tuning.profile import _base_fingerprint, _hash_fingerprint
from freetoken.utils import init_logger
from freetoken.version import __version__

logger = init_logger(__name__)

_ENABLED_ENV = "FREETOKEN_DIAGNOSTICS"

# Process-wide attribution, set once at engine boot. A single GPU/model per serving process, so
# a module-level context is the simplest correct thing (no threading of model_path through every
# call site -- moe/offload_cache and the memory planner have no reference to it).
_key: str | None = None
_model_path: str | None = None
_config_summary: dict[str, Any] = {}


def _enabled() -> bool:
    return (os.getenv(_ENABLED_ENV) or "1").strip().lower() not in ("0", "false", "no", "off")


def _gpu_uuid() -> str:
    try:
        from freetoken.gpu_select import _nvml_uuids

        uuids = _nvml_uuids()
    except Exception:  # noqa: BLE001 -- best-effort; a broken NVML load must never block boot
        return "unknown-gpu"
    return uuids[0] if uuids else "unknown-gpu"


def diagnostics_dir() -> str:
    return os.path.join(_cache_dir(), "diagnostics")


def diagnostics_path(model_path: str | None = None) -> str | None:
    """The JSONL for this machine's first GPU + ``model_path`` (default: the configured one)."""
    mp = model_path or _model_path
    if not mp:
        return None
    key = _hash_fingerprint(_base_fingerprint(gpu_uuid=_gpu_uuid(), model_path=mp))
    return os.path.join(diagnostics_dir(), f"{key}.jsonl")


def configure(model_path: str, config_summary: dict[str, Any] | None = None) -> str | None:
    """Attribute subsequent events to ``model_path``'s fingerprint and stamp them with a config
    summary (spec_mtp, kv_format, max_seq_len, moe_strategy, ...). Returns the diagnostics path,
    or None when disabled. Best-effort; never raises."""
    global _key, _model_path, _config_summary
    try:
        _model_path = model_path
        _config_summary = dict(config_summary or {})
        _config_summary.setdefault("freetoken_version", __version__)
        if not _enabled():
            _key = None
            return None
        _key = _hash_fingerprint(_base_fingerprint(gpu_uuid=_gpu_uuid(), model_path=model_path))
        os.makedirs(diagnostics_dir(), exist_ok=True)
        return os.path.join(diagnostics_dir(), f"{_key}.jsonl")
    except Exception as e:  # noqa: BLE001 -- diagnostics must never break boot
        logger.info(f"diagnostics disabled (configure failed: {e})")
        _key = None
        return None


def log_event(
    kind: str,
    phase: str,
    message: str,
    *,
    severity: str = "warn",
    **context: Any,
) -> None:
    """Append one structured event. ``kind`` is a coarse bucket (oom | degrade | fallback |
    error | profile), ``phase`` names where it happened, ``message`` is the human line (the same
    one already going to the server log), and ``context`` carries numbers worth keeping (free
    VRAM, requested bytes, the degraded value). NEVER raises; drops silently if unconfigured or
    disabled -- a diagnostics write must not take down serving."""
    # Always mirror to the normal logger so the live server log has it too.
    line = f"[diag:{kind}/{phase}] {message}"
    (logger.warning if severity in ("warn", "error") else logger.info)(line)
    if not _enabled() or _key is None:
        return
    try:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kind": kind,
            "phase": phase,
            "severity": severity,
            "message": message,
            "config": _config_summary,
            "context": context or {},
        }
        path = os.path.join(diagnostics_dir(), f"{_key}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:  # noqa: BLE001 -- swallow; diagnostics are best-effort
        pass


def vram_snapshot(device: Any = None) -> dict[str, Any]:
    """A small, allocation-free VRAM snapshot for event context (best-effort)."""
    try:
        import torch

        if device is None:
            device = torch.device("cuda")
        free, total = torch.cuda.mem_get_info(device)
        return {
            "free_mib": round(free / (1 << 20), 1),
            "total_mib": round(total / (1 << 20), 1),
            "allocated_mib": round(torch.cuda.memory_allocated(device) / (1 << 20), 1),
            "reserved_mib": round(torch.cuda.memory_reserved(device) / (1 << 20), 1),
        }
    except Exception:  # noqa: BLE001
        return {}
