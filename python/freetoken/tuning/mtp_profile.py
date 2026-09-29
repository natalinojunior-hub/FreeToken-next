"""Persisted MTP speculation-depth profile: the depth the adaptive controller learned to be
optimal, keyed on the hardware + model + build + serve-config fingerprint it was measured
under.

Runtime sibling of ``freetoken.tuning.profile`` (the ``ft tune`` end-to-end auto-config
profile). That one tunes the ``spec_mtp`` *cap* via a deliberate measurement run; this one
caches the depth the *adaptive controller* converges to at runtime (0..cap), so the next serve
of the same model on the same machine skips the 24-cycle calibration probe and starts at the
learned depth directly. On the campaign37 cert model that is the difference between ~104 TG
(warm, no probe overhead) and ~98-101 TG (paying the probe every cold start).

VERSIONING + FORCED RECALIBRATION (the operator requirement this module exists to satisfy):

* The key folds in a *content* fingerprint of every source tree that can change which depth
  wins -- ``kernel/`` + ``moe/`` (numerics and timing) and ``scheduler/`` + ``engine/`` (the
  depth-selection logic: the adaptive controller, the spec driver, the resolver). Editing any
  file under them changes the fingerprint, so a stale profile is never silently trusted: the
  engine recalibrates and re-saves. This is the automatic, fine-grained version, and it needs
  no manual ``__version__`` bump (``__version__`` is coupled to the installed kernel-cache
  package's ABI check, so bumping it in isolation would break the build).
* ``__version__`` (coarse release version) and ``SCHEMA_VERSION`` (this file's format) are
  also in the key.
* ``FREETOKEN_MTP_PROFILE`` is the manual control: ``auto`` (default -- load if valid, save on
  convergence), ``off`` (never load or save -- pure per-run adaptation), ``refresh`` (ignore
  any cached profile this run, recalibrate, and overwrite -- a forced one-shot recalibration).

A schema/key mismatch is ignored with a single log line and falls back to recalibrating --
never trusted blindly, exactly like ``tuning.profile``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from freetoken.moe.bench_profile import _cache_dir
from freetoken.tuning.profile import _ceil_pow2, _model_identity
from freetoken.utils import init_logger
from freetoken.version import __version__

logger = init_logger(__name__)

# Bump when the on-disk schema changes; a stored profile with a different schema is ignored
# (recalibrated) rather than mis-read.
SCHEMA_VERSION = 1

_PROFILE_MODE_ENV = "FREETOKEN_MTP_PROFILE"
_MODE_AUTO = "auto"
_MODE_OFF = "off"
_MODE_REFRESH = "refresh"
_VALID_MODES = (_MODE_AUTO, _MODE_OFF, _MODE_REFRESH)

# Source trees whose CONTENT can change which MTP depth is optimal. Hashed into the key so any
# edit to the depth-deciding logic forces recalibration (see the module docstring).
_CALIBRATION_SOURCE_BASES = ("kernel", "moe", "scheduler", "engine")
_SOURCE_SUFFIXES = (".py", ".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp")


def _profile_mode() -> str:
    raw = (os.getenv(_PROFILE_MODE_ENV) or _MODE_AUTO).strip().lower()
    return raw if raw in _VALID_MODES else _MODE_AUTO


@lru_cache(maxsize=1)
def _calibration_source_fingerprint() -> str:
    """Content hash of the depth-deciding source trees, cached per process (sources do not
    change at runtime). Content, not mtime, so a fresh checkout of identical sources keys the
    same while any real edit invalidates every stored profile."""
    root = os.path.dirname(os.path.dirname(__file__))  # python/freetoken
    h = hashlib.sha256()
    paths: list[str] = []
    for base in _CALIBRATION_SOURCE_BASES:
        for dirpath, _dirs, files in os.walk(os.path.join(root, base)):
            paths += [
                os.path.join(dirpath, n) for n in files if n.endswith(_SOURCE_SUFFIXES)
            ]
    for p in sorted(paths, key=lambda p: os.path.relpath(p, root)):
        try:
            with open(p, "rb") as f:
                data = f.read()
        except OSError:
            continue
        h.update(os.path.relpath(p, root).encode() + b"\0" + data)
    return h.hexdigest()[:16]


def _gpu_uuid() -> str:
    try:
        from freetoken.gpu_select import _nvml_uuids

        uuids = _nvml_uuids()
    except Exception:  # noqa: BLE001 -- best-effort; a broken NVML load must never block boot
        return "unknown-gpu"
    return uuids[0] if uuids else "unknown-gpu"


def _profile_dir() -> str:
    return os.path.join(_cache_dir(), "mtp_depth")


def profile_path(key: str) -> str:
    return os.path.join(_profile_dir(), f"{key}.json")


def compute_key(
    *,
    model_path: str,
    kv_format: str,
    max_seq_len: int,
    spec_mtp_cap: int,
    max_running_req: int,
    moe_strategy: str,
    compact_state: bool,
    ssm_dtype: str,
) -> str:
    """The lookup key: everything that can change which depth wins.

    Hardware + model + build identity (GPU uuid, model weights, engine version, and the
    calibration-source content hash) plus the serve-config axes that shift depth economics:
    the spec cap, batch size, MoE strategy, KV format, a context-length bucket (ceil pow2),
    and the two residency reducers the engine resolver may have auto-enabled or the caller may
    have pinned (compact verify state, SSM dtype). Any change invalidates the entry -- a fresh
    key, no file, recalibrate.
    """
    fingerprint: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "gpu_uuid": _gpu_uuid(),
        "model_id": _model_identity(model_path),
        "freetoken_version": __version__,
        "calibration_source": _calibration_source_fingerprint(),
        "kv_format": kv_format,
        "ctx_bucket": _ceil_pow2(max(1, int(max_seq_len))),
        "spec_mtp_cap": int(spec_mtp_cap),
        "max_running_req": int(max_running_req),
        "moe_strategy": moe_strategy,
        "compact_state": bool(compact_state),
        "ssm_dtype": str(ssm_dtype).lower(),
    }
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:24]


def key_from_config(config, effective_max_seq_len: int, spec_mtp_cap: int) -> str:
    """Build the key from an ``EngineConfig`` after the engine resolver has run (so the
    compact-state env and ``ENV.MAMBA_SSM_DTYPE`` already reflect the auto-enabled values)."""
    from freetoken.env import ENV

    return compute_key(
        model_path=config.model_path,
        kv_format=str(getattr(config, "kv_format", "auto")),
        max_seq_len=effective_max_seq_len,
        spec_mtp_cap=spec_mtp_cap,
        max_running_req=int(getattr(config, "max_running_req", 1)),
        moe_strategy=str(getattr(config, "moe_strategy", "")),
        compact_state=os.getenv("FREETOKEN_MTP_COMPACT_STATE", "0") == "1",
        ssm_dtype=str(ENV.MAMBA_SSM_DTYPE),
    )


@dataclass
class DepthEvidence:
    """Informational only -- the decision uses the key match and ``depth``, never this."""

    date: str = ""
    committed_tg: float | None = None
    note: str = ""


@dataclass
class DepthProfile:
    key: str
    depth: int
    evidence: DepthEvidence
    schema: int = SCHEMA_VERSION

    def to_json(self) -> dict:
        return {
            "schema": self.schema,
            "key": self.key,
            "depth": self.depth,
            "evidence": asdict(self.evidence),
        }


def save(key: str, depth: int, evidence: DepthEvidence | None = None) -> str | None:
    """Atomically persist the learned ``depth``. Returns the path, or ``None`` when caching is
    disabled (``off``) or the write fails -- a profile write must never block serving."""
    if _profile_mode() == _MODE_OFF:
        return None
    if evidence is None:
        evidence = DepthEvidence()
    if not evidence.date:
        evidence.date = datetime.now(timezone.utc).isoformat(timespec="seconds")
    dest = profile_path(key)
    profile = DepthProfile(key=key, depth=int(depth), evidence=evidence)
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = f"{dest}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(profile.to_json(), f, indent=2)
        os.replace(tmp, dest)
    except OSError as e:
        logger.info(f"mtp depth profile {dest}: write failed ({e}); continuing without cache")
        return None
    logger.info(f"mtp depth profile: saved depth={int(depth)} -> {dest}")
    return dest


def load(key: str) -> int | None:
    """The cached optimal depth for ``key``, or ``None`` when there is no file, it fails to
    parse, its schema/key does not match, or the mode disables loading (``off``/``refresh``).
    Every rejection logs one line -- a silently-ignored stale profile is confusing."""
    mode = _profile_mode()
    if mode == _MODE_OFF:
        return None
    if mode == _MODE_REFRESH:
        logger.info("mtp depth profile: FREETOKEN_MTP_PROFILE=refresh -> forced recalibration")
        return None
    src = profile_path(key)
    try:
        with open(src) as f:
            raw = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        logger.info(f"mtp depth profile {src}: unreadable ({e}); recalibrating")
        return None
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA_VERSION:
        got = raw.get("schema") if isinstance(raw, dict) else "?"
        logger.info(f"mtp depth profile {src}: schema {got} != {SCHEMA_VERSION}; recalibrating")
        return None
    if raw.get("key") != key:
        logger.info(f"mtp depth profile {src}: key mismatch (stale/foreign); recalibrating")
        return None
    depth = raw.get("depth")
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 0:
        logger.info(f"mtp depth profile {src}: malformed depth {depth!r}; recalibrating")
        return None
    logger.info(f"mtp depth profile: warm start at depth={depth} from {src} (no calibration probe)")
    return depth
