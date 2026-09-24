"""Opt-in bounded token trace for correctness investigations."""

from __future__ import annotations

import atexit
import json
import os
import signal
from pathlib import Path
from typing import Any

_path = os.environ.get("FREETOKEN_TOKEN_TRACE")
_records: list[dict[str, Any]] = []
_limit = 4096


def enabled() -> bool:
    return bool(_path)


def record(**fields: Any) -> None:
    if not _path or len(_records) >= _limit:
        return
    _records.append(fields)


def flush() -> None:
    if not _path or not _records:
        return
    path = Path(_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for item in _records:
            stream.write(json.dumps(item, separators=(",", ":")) + "\n")
    _records.clear()


atexit.register(flush)


def _flush_on_term(_signum: int, _frame: Any) -> None:
    flush()
    raise SystemExit(0)


if _path:
    signal.signal(signal.SIGTERM, _flush_on_term)
