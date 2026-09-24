#!/usr/bin/env python3
"""Summarize FREETOKEN_MOE_EVENTS output and interval unions."""

import json
import sys


def main(path: str) -> None:
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    by_label: dict[str, list[tuple[float, float]]] = {}
    for item in data.get("intervals", []):
        by_label.setdefault(item["label"], []).append((item["start_ms"], item["end_ms"]))
    for label, intervals in sorted(by_label.items()):
        total = sum(max(0.0, end - start) for start, end in intervals)
        union = 0.0
        cursor = -1.0
        for start, stop in sorted(intervals):
            if start > cursor:
                union += max(0.0, stop - start)
            else:
                union += max(0.0, stop - cursor)
            cursor = max(cursor, stop)
        print(f"{label}: samples={len(intervals)} sum_ms={total:.3f} union_ms={union:.3f}")
    print(
        f"event_count={data.get('event_count', 0)} record_overhead_ms={data.get('record_overhead_ms', 0.0):.3f}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: analyze_moe_events.py EVENTS.json")
    main(sys.argv[1])
