"""Persisted ``ft tune`` runtime profile: chosen v1 auto-config settings, keyed on the
hardware + model + build combination they were measured under.

Separate from ``freetoken.moe.bench_profile`` (a synthetic, format-only CPU-vs-PCIe
bandwidth profile that feeds the MoE offload/hybrid *default*): this profile is an
end-to-end measurement (real server boots, real cold PP / committed TG / VRAM) over the
full v1 tunable set (``spec_mtp``, ``defer_replay``, ``draft_graph``, ``moe_strategy``,
``cpu_threads``), written by ``ft tune`` and read at arg-resolution time.

Schema is versioned; a mismatched key or an old/foreign schema version is ignored (falls
back to whatever the caller's own defaults are) with one log line -- never trusted blindly.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from freetoken.moe.bench_profile import _cache_dir
from freetoken.utils import init_logger
from freetoken.version import __version__

logger = init_logger(__name__)

SCHEMA_VERSION = 1

# v1 tunables this profile stores a chosen value for.
TUNABLE_FIELDS = ("spec_mtp", "defer_replay", "draft_graph", "moe_strategy", "cpu_threads")


def _profile_dir() -> str:
    return os.path.join(_cache_dir(), "tune")


def profile_path(key: str) -> str:
    return os.path.join(_profile_dir(), f"{key}.json")


def _ceil_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _kernel_source_fingerprint() -> str:
    """Content hash of the kernel/moe source trees: a changed kernel can change the numerics
    or timing a candidate was measured under. Contents, not mtimes, so a fresh checkout or a
    second worktree of the same sources keys identically."""
    root = os.path.dirname(os.path.dirname(__file__))  # python/freetoken
    h = hashlib.sha256()
    paths = []
    for base in (os.path.join(root, "kernel"), os.path.join(root, "moe")):
        for dirpath, _dirs, files in os.walk(base):
            paths += [
                os.path.join(dirpath, n)
                for n in files
                if n.endswith((".py", ".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp"))
            ]
    for p in sorted(paths, key=lambda p: os.path.relpath(p, root)):
        try:
            with open(p, "rb") as f:
                data = f.read()
        except OSError:
            continue
        h.update(os.path.relpath(p, root).encode() + b"\0" + data)
    return h.hexdigest()[:16]


def _model_identity(model_path: str) -> str:
    """Resolved path + total size + mtime of every weight file under it, hashed -- cheap
    (stat only, no reads) and changes if the checkpoint is swapped or re-converted."""
    resolved = os.path.realpath(model_path)
    entries: list[tuple[str, int, float]] = []
    if os.path.isdir(resolved):
        for dirpath, _dirs, files in os.walk(resolved):
            for name in files:
                if not name.endswith((".safetensors", ".gguf", ".ftw")):
                    continue
                p = os.path.join(dirpath, name)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                entries.append((os.path.relpath(p, resolved), st.st_size, st.st_mtime))
    elif os.path.isfile(resolved):
        st = os.stat(resolved)
        entries.append((os.path.basename(resolved), st.st_size, st.st_mtime))
    entries.sort()
    payload = repr((resolved, entries)).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def compute_key(
    *,
    gpu_uuid: str,
    model_path: str,
    kv_format: str,
    max_seq_len: int,
) -> str:
    """The profile lookup key: everything that can change which candidate wins.

    GPU (bandwidth/VRAM), model (weights + geometry), kv_format (attention backend/memory
    shape), a context-length bucket (ceil pow2 -- exact length doesn't matter, only which
    power-of-two regime), the FreeToken version, and a kernel-source content
    fingerprint (a rebuilt kernel can change the numerics/timing a candidate was measured
    under). Any change to any of these invalidates the entry (a fresh key -> no file ->
    caller falls back to its own default).
    """
    fingerprint = {
        "schema": SCHEMA_VERSION,
        "gpu_uuid": gpu_uuid,
        "model_id": _model_identity(model_path),
        "kv_format": kv_format,
        "ctx_bucket": _ceil_pow2(max(1, max_seq_len)),
        "freetoken_version": __version__,
        "kernel_fingerprint": _kernel_source_fingerprint(),
    }
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:24]


@dataclass
class CandidateEvidence:
    cold_pp: float
    committed_tg: float
    ttft_s: float
    peak_vram_mib: float
    date: str


@dataclass
class TunedSettings:
    spec_mtp: int
    defer_replay: bool
    draft_graph: bool
    moe_strategy: str  # "offload" | "hybrid"
    cpu_threads: int = 0


@dataclass
class Profile:
    schema: int
    key: str
    chosen: TunedSettings
    evidence: CandidateEvidence
    candidates: list[dict[str, Any]] = field(default_factory=list)  # every measured candidate

    def to_json(self) -> dict:
        return {
            "schema": self.schema,
            "key": self.key,
            "chosen": asdict(self.chosen),
            "evidence": asdict(self.evidence),
            "candidates": self.candidates,
        }


def save(profile: Profile, path: str | None = None) -> str:
    dest = path or profile_path(profile.key)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = f"{dest}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(profile.to_json(), f, indent=2)
    os.replace(tmp, dest)
    return dest


def load(key: str, path: str | None = None) -> Profile | None:
    """The profile for ``key``, or ``None`` when there is no file, it fails to parse, its
    schema doesn't match, or its own key doesn't match ``key`` (a hand-copied/renamed file).
    Every rejection path logs one line -- a silently-ignored stale profile is confusing."""
    src = path or profile_path(key)
    try:
        with open(src) as f:
            raw = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        logger.info(f"tune profile {src}: unreadable ({e}); ignoring, using defaults")
        return None
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA_VERSION:
        logger.info(
            f"tune profile {src}: schema {raw.get('schema') if isinstance(raw, dict) else '?'} "
            f"!= {SCHEMA_VERSION}; ignoring, using defaults"
        )
        return None
    if raw.get("key") != key:
        logger.info(f"tune profile {src}: key mismatch (stale/foreign hardware); ignoring")
        return None
    try:
        chosen = TunedSettings(**raw["chosen"])
        evidence = CandidateEvidence(**raw["evidence"])
    except (KeyError, TypeError) as e:
        logger.info(f"tune profile {src}: malformed ({e}); ignoring, using defaults")
        return None
    return Profile(
        schema=raw["schema"],
        key=raw["key"],
        chosen=chosen,
        evidence=evidence,
        candidates=raw.get("candidates", []),
    )


def load_for(model_path: str, kv_format: str, max_seq_len: int | None) -> Profile | None:
    """The stored profile for this machine's first GPU and this serve configuration, or
    ``None``. NVML, not torch.cuda: callers run in the parent before the scheduler spawns."""
    if max_seq_len is None:
        return None
    try:
        from freetoken.gpu_select import _nvml_uuids

        uuids = _nvml_uuids()
    except Exception:  # noqa: BLE001 -- best-effort; a broken NVML load must never block boot
        return None
    if not uuids:
        return None
    key = compute_key(
        gpu_uuid=uuids[0], model_path=model_path, kv_format=kv_format, max_seq_len=max_seq_len
    )
    return load(key)


def env_overrides(settings: TunedSettings) -> dict[str, str]:
    """The env-backed v1 tunables' values for ``settings``, as the env-var strings
    ``scheduler/spec.py`` / ``engine/graph.py`` read (``os.getenv(..., "1"/"0") != "0"``)."""
    return {
        "FREETOKEN_SPEC_DEFER_REPLAY": "1" if settings.defer_replay else "0",
        "FREETOKEN_DRAFT_GRAPH": "1" if settings.draft_graph else "0",
    }


def apply_env_defaults(settings: TunedSettings, environ: dict) -> list[str]:
    """setdefault each env-backed tunable into ``environ`` (only if the key is absent --
    an explicit user-set value, "0" or "1", is never touched); returns the keys actually
    set, for a one-line log at the call site.

    Called from ``server/launch.py``'s ``apply_tuning_profile_env_gate``, in the parent,
    before the scheduler subprocess spawns (NVML for the GPU uuid, not ``torch.cuda`` --
    no CUDA context in the parent before ``mp.Process(..., start_method="spawn")``).
    """
    applied = []
    for k, v in env_overrides(settings).items():
        if k not in environ:
            environ[k] = v
            applied.append(k)
    return applied


_UNSET = object()


def resolve(user_value: Any, sentinel: Any, tuned_value: Any) -> Any:
    """Precedence rule shared by every v1 tunable: an explicit user value (CLI flag or env
    var the user actually set) always wins; only a value left at its "auto"/default
    sentinel is filled from the tuned profile. No profile (``tuned_value is _UNSET`` or
    ``None``) leaves the sentinel as-is -- the caller's own existing default applies."""
    if user_value != sentinel:
        return user_value
    if tuned_value is None or tuned_value is _UNSET:
        return user_value
    return tuned_value
