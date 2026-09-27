#!/usr/bin/env python3
"""Replay an opt-in ``FREETOKEN_MOE_TRACE`` JSONL capture."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _row_bytes(row: dict) -> float:
    return row["miss_bytes"] / row["missing"] if row["missing"] else 0.0


def actual_replay(rows: list[dict]) -> tuple[int, float, int]:
    misses = copied = mismatches = 0
    for row in rows:
        ids = list(row["before_ids"])
        usage = list(row["before_usage"])
        requested = list(dict.fromkeys(row["requested_global_ids"]))
        missing = sorted(expert for expert in requested if expert not in ids)
        step = max(row["after_usage"], default=max(usage, default=0) + 1)
        for expert in requested:
            if expert in ids:
                usage[ids.index(expert)] = step
        victims = sorted(range(len(ids)), key=lambda slot: (usage[slot], slot))[: len(missing)]
        for expert, slot in zip(missing, victims):
            ids[slot] = expert
        mismatches += ids != row["after_ids"]
        misses += len(missing)
        copied += len(missing) * _row_bytes(row)
    return misses, copied, mismatches


def true_lru(rows: list[dict]) -> tuple[int, float]:
    state: dict[int, tuple[list[int], list[int]]] = {}
    clock = 0
    misses = copied = 0.0
    for row in rows:
        pool = row["pool"]
        if pool not in state:
            state[pool] = (list(row["before_ids"]), list(row["before_usage"]))
        ids, usage = state[pool]
        row_bytes = _row_bytes(row)
        requested = list(dict.fromkeys(row["requested_global_ids"]))
        clock += 1
        for expert in requested:
            if expert in ids:
                usage[ids.index(expert)] = clock
        for expert in requested:
            if expert in ids:
                continue
        missing = [expert for expert in requested if expert not in ids]
        victims = sorted(range(len(ids)), key=lambda i: (usage[i], i))[: len(missing)]
        for expert, slot in zip(missing, victims):
            ids[slot] = expert
            usage[slot] = clock
            misses += 1
            copied += row_bytes
    return int(misses), copied


def belady(rows: list[dict]) -> tuple[int, float]:
    future: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        for expert in set(row["requested_global_ids"]):
            future[(row["pool"], expert)].append(index)
    state: dict[int, list[int]] = {}
    misses = copied = 0.0
    for index, row in enumerate(rows):
        pool = row["pool"]
        ids = state.setdefault(pool, list(row["before_ids"]))
        row_bytes = _row_bytes(row)
        requested = list(dict.fromkeys(row["requested_global_ids"]))
        for expert in requested:
            future[(pool, expert)].pop(0)
        for expert in requested:
            if expert in ids:
                continue
            empty = next((slot for slot, value in enumerate(ids) if value < 0), None)
            if empty is not None:
                slot = empty
            else:
                slot = max(
                    range(len(ids)),
                    key=lambda i: (
                        future.get((pool, ids[i]), [10**9])[0]
                        if future.get((pool, ids[i]))
                        else 10**9,
                        i,
                    ),
                )
            ids[slot] = expert
            misses += 1
            copied += row_bytes
    return int(misses), copied


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    rows = _events(args.trace)
    actual = actual_replay(rows)
    lru = true_lru(rows)
    oracle = belady(rows)
    print(json.dumps({"records": len(rows), "actual": actual, "true_lru": lru, "belady": oracle}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
