"""Persisted decode-peak VRAM learning: the allocation size that once refused a decode
forward is reserved (and kept reserved for growth headroom) on every later serve of the
same fingerprint. Sibling of ``mtp_profile`` but keyed with two extra axes -- vision
(mmproj encoders resident) and the MTP draft head -- because both change the per-step
transient footprint, and the operator requirement is that ANY model/quantization/KV-format
combination self-adapts with zero manual guard tuning (fp8, nvfp4, turbo3/4/8, dense or
MoE, with or without mmproj/MTP).

The base key reuses ``mtp_profile.key_from_config`` (GPU uuid, model weights, engine
build, calibration-source hash, kv_format, max_seq_len, spec cap, running-reqs, MoE
strategy, compact state, ssm dtype); the VRAM key folds the extra axes on top. Learned
bytes never decrease (a smaller peak is already covered). Load/save are best-effort: a
missing or unreadable profile simply starts the learning from zero.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from freetoken.tuning.mtp_profile import key_from_config, profile_path

from freetoken.tuning.mtp_profile import _profile_dir  # noqa: F401  (cache-root reuse)

SCHEMA = "vram1"


def compute_key(
    config, effective_max_seq_len: int, spec_mtp_cap: int, has_vision: bool, has_mtp_head: bool
) -> str:
    # No source hash: Phase I measures every cached calibration in place and re-solves on a
    # mismatch, so an unrelated commit must not cost a full Phase D re-calibration.
    base = key_from_config(config, effective_max_seq_len, spec_mtp_cap, include_source=False)
    # Attention backend changes transient workspace and graph geometry.  Keep it in the
    # fingerprint so a profile learned by Triton cannot silently warm-start another backend.
    attention = str(getattr(config, "attention_backend", ""))
    axes = f"{SCHEMA}|{base}|a={attention}|v={int(bool(has_vision))}|m={int(bool(has_mtp_head))}"
    return hashlib.sha1(axes.encode()).hexdigest()[:24]


def _path(key: str) -> str:
    d = os.path.join(os.path.dirname(profile_path(key)), "vram_learned")
    return os.path.join(d, f"{key}.json")


def _write(key: str, data: dict[str, Any]) -> None:
    path = _path(key)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except OSError:
        pass  # profile persistence must never block serving


def save(key: str, learned_bytes: int) -> None:
    if learned_bytes <= 0:
        return
    try:
        with open(_path(key)) as f:
            data: Any = json.load(f)
    except (OSError, ValueError, TypeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.update(schema=SCHEMA, learned_bytes=int(learned_bytes))
    _write(key, data)


def load(key: str) -> int:
    try:
        with open(_path(key)) as f:
            data: Any = json.load(f)
        if data.get("schema") != SCHEMA:
            return 0
        value = int(data.get("learned_bytes", 0))
        return value if value > 0 else 0
    except (OSError, ValueError, TypeError):
        return 0


def save_calibration(key: str, calibration: Any, *, geometry: dict[str, Any]) -> None:
    """Persist measured planner calibration; callers keep physical validation in the planner."""
    try:
        with open(_path(key)) as f:
            data: Any = json.load(f)
    except (OSError, ValueError, TypeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    values = {
        name: max(0, int(getattr(calibration, name)))
        for name in (
            "chunk_lo",
            "transient_lo",
            "chunk_hi",
            "transient_hi",
            "lazy_persistent",
            "graph_capture_peak",
            "graph_pool_size",
            "non_pytorch_growth",
        )
    }
    previous = data.get("calibration")
    history = data.get("calibration_history", [])
    if not isinstance(history, list):
        history = []
    samples = int(data.get("calibration_samples", 0)) if isinstance(data, dict) else 0
    # A single boot can be disturbed by another process.  Require two matching successful
    # planner observations before the fast path trusts the measurement.
    if isinstance(previous, dict):
        deltas = [abs(int(previous.get(k, 0)) - values[k]) for k in values]
        if any(delta > max(4 << 20, values[k] // 10) for k, delta in zip(values, deltas)):
            samples = 0
            history = []
    history.append(values)
    history = history[-5:]
    data.update(
        schema=SCHEMA,
        calibration=values,
        geometry=dict(geometry),
        calibration_history=history,
        calibration_samples=max(samples + 1, len(history)),
    )
    _write(key, data)


def load_calibration(
    key: str, *, driver_total: int, page_size: int, attention_backend: str
) -> dict[str, Any] | None:
    """Return a calibration only when its physical geometry still matches this serve."""
    try:
        with open(_path(key)) as f:
            data: Any = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return None
    geometry = data.get("geometry")
    calibration = data.get("calibration")
    history = data.get("calibration_history", [])
    if (
        not isinstance(geometry, dict)
        or not isinstance(calibration, dict)
        or not isinstance(history, list)
        or int(data.get("calibration_samples", 0)) < 2
    ):
        return None
    saved_total = int(geometry.get("driver_total", 0))
    # Permit small driver-reporting drift, but never reuse a plan across a materially
    # different device or page/attention geometry.
    if saved_total <= 0 or abs(saved_total - int(driver_total)) > max(64 << 20, saved_total // 100):
        return None
    if int(geometry.get("page_size", 0)) != int(page_size):
        return None
    if str(geometry.get("attention_backend", "")) != str(attention_backend):
        return None
    fields = (
        "chunk_lo",
        "transient_lo",
        "chunk_hi",
        "transient_hi",
        "lazy_persistent",
        "graph_capture_peak",
        "graph_pool_size",
        "non_pytorch_growth",
    )
    try:
        samples = [
            {name: int(sample[name]) for name in fields}
            for sample in history
            if isinstance(sample, dict)
        ]
        if len(samples) < 2:
            samples = [{name: int(calibration[name]) for name in fields}]
        result = {
            name: sorted(sample[name] for sample in samples)[len(samples) // 2] for name in fields
        }
    except (KeyError, TypeError, ValueError):
        return None
    if result["chunk_lo"] <= 0 or result["chunk_hi"] < result["chunk_lo"]:
        return None
    return result
