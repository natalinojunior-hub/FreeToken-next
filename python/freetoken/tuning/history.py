"""Persisted per-model run history: PP/TG/KV-placement measurements from real benchmark
runs, keyed on hardware + model + build. Written by ``benchmarks/bench_pp_tg.py``, read by
future context-advisor tooling, so a later run against the same model can see what was
measured before. JSON Lines: one record per configuration, oldest first.

Reuses ``freetoken.tuning.profile``'s hardware+model fingerprint (GPU uuid, model weight
identity, FreeToken version, kernel-source content hash) and cache dir -- do not duplicate
that hashing logic here.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any

from freetoken.moe.bench_profile import _cache_dir
from freetoken.tuning.profile import _base_fingerprint, _hash_fingerprint
from freetoken.utils import init_logger
from freetoken.version import __version__

logger = init_logger(__name__)

SCHEMA_VERSION = 1


def _history_dir() -> str:
    return os.path.join(_cache_dir(), "tune", "history")


def _gpu_uuid() -> str:
    try:
        from freetoken.gpu_select import _nvml_uuids

        uuids = _nvml_uuids()
    except Exception:  # noqa: BLE001 -- best-effort; a broken NVML load must never block a run
        return "unknown-gpu"
    return uuids[0] if uuids else "unknown-gpu"


def _history_key(model_path: str) -> str:
    return _hash_fingerprint(_base_fingerprint(gpu_uuid=_gpu_uuid(), model_path=model_path))


def history_path(model_path: str) -> Path:
    return Path(_history_dir()) / f"{_history_key(model_path)}.jsonl"


def append_run(model_path: str, record: dict[str, Any]) -> Path:
    """Append one run record to ``model_path``'s history file. Fills in ``schema``,
    ``fingerprint_key``, ``freetoken_version`` and a UTC ``timestamp`` on the record.
    Opens with O_APPEND and writes the line in one syscall, so concurrent appenders from
    separate processes interleave whole lines rather than corrupting each other. Never
    raises into the caller -- an I/O error is logged and swallowed, since a failed history
    write must never fail the benchmark run it's recording."""
    dest = history_path(model_path)
    full = {
        **record,
        "schema": SCHEMA_VERSION,
        "fingerprint_key": dest.stem,
        "freetoken_version": __version__,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    line = (json.dumps(full, sort_keys=True) + "\n").encode()
    try:
        os.makedirs(dest.parent, exist_ok=True)
        fd = os.open(dest, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except OSError as e:
        logger.warning(f"history append to {dest}: failed ({e}); run record lost")
    return dest


def load_runs(model_path: str) -> list[dict[str, Any]]:
    """Every record in ``model_path``'s history file with a matching schema. A record with
    a foreign/old schema, or an unparseable line, is skipped with one log line rather than
    trusted blindly."""
    src = history_path(model_path)
    try:
        with open(src) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []
    except OSError as e:
        logger.warning(f"history read {src}: failed ({e}); treating as empty")
        return []
    runs = []
    for i, raw_line in enumerate(lines):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            rec = json.loads(raw_line)
        except ValueError as e:
            logger.info(f"history {src}:{i + 1}: unparseable ({e}); skipping")
            continue
        if not isinstance(rec, dict) or rec.get("schema") != SCHEMA_VERSION:
            got = rec.get("schema") if isinstance(rec, dict) else "?"
            logger.info(f"history {src}:{i + 1}: schema {got} != {SCHEMA_VERSION}; skipping")
            continue
        runs.append(rec)
    return runs
