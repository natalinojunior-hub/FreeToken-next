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
    base = key_from_config(config, effective_max_seq_len, spec_mtp_cap)
    axes = f"{SCHEMA}|{base}|v={int(bool(has_vision))}|m={int(bool(has_mtp_head))}"
    return hashlib.sha1(axes.encode()).hexdigest()[:24]


def _path(key: str) -> str:
    d = os.path.join(os.path.dirname(profile_path(key)), "vram_learned")
    return os.path.join(d, f"{key}.json")


def save(key: str, learned_bytes: int) -> None:
    if learned_bytes <= 0:
        return
    path = _path(key)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"schema": SCHEMA, "learned_bytes": int(learned_bytes)}, f)
        os.replace(tmp, path)
    except OSError:
        pass  # profile persistence must never block serving


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
