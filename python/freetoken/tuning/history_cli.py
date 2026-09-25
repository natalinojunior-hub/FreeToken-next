"""``ft history <model>``: print this model's persisted run-history log (see
``history.py`` for the schema/fingerprint), newest run first, as a compact table.
"""

from __future__ import annotations

import argparse
import sys

from freetoken.tuning.history import load_runs


def _fmt(v, spec: str = "") -> str:
    if v is None:
        return "-"
    if spec:
        try:
            return format(v, spec)
        except (ValueError, TypeError):
            return str(v)
    return str(v)


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    p = argparse.ArgumentParser(prog=prog, description=__doc__)
    p.add_argument("model", help="checkpoint dir / .ftw dir / .gguf path")
    p.add_argument("--limit", type=int, default=20, help="most-recent runs to show (default 20)")
    args = p.parse_args(argv)

    runs = load_runs(args.model)
    if not runs:
        print(f"sem histórico para {args.model}")
        return 0

    rows = list(reversed(runs))[: args.limit]
    header = (
        f"{'data':<20} {'rótulo':<12} {'contexto':>8} {'PP':>8} {'TG':>7} "
        f"{'experts':>7} {'KV GPU/RAM páginas':>19} {'hash':<12}"
    )
    print(header)
    for r in rows:
        date = str(r.get("timestamp", "-"))[:19]
        kv = f"{_fmt(r.get('kv_device_pages'))}/{_fmt(r.get('kv_ram_pages'))}"
        print(
            f"{date:<20} {_fmt(r.get('label')):<12} {_fmt(r.get('tokens')):>8} "
            f"{_fmt(r.get('pp_tok_s_mean'), '.1f'):>8} {_fmt(r.get('tg_tok_s_mean'), '.2f'):>7} "
            f"{_fmt(r.get('expert_slots')):>7} {kv:>19} {_fmt(r.get('output_sha1')):<12}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], prog="ft history"))
