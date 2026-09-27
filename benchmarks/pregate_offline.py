#!/usr/bin/env python3
"""Offline pre-gating recall gate (decisions/pre-gating-prefetch-design steps 2-3).

Inputs: a FREETOKEN_MOE_TRACE capture dir (trace.jsonl + snapshot.json + hidden.pt
from FREETOKEN_MOE_TRACE_HIDDEN=1) and the GGUF checkpoint (router weights
``blk.N.ffn_gate_inp.weight``). Evaluates cheap next-layer expert predictors
against the real routing and replays the LRU pools with one-layer-early
prefetch installed, reporting exposed miss bytes, pollution, contention and
projected net ms/token. CPU-only.

Predictors (issued at layer L's router time, targeting the next executed MoE
layer L+1, wrapping across the step boundary):
  P0 persistence : actual ids of layer L (free)
  P1 early-router: topk'(W_gate[L+1] @ h_L) using the REAL next-layer router
                   weights on layer L's MoE input (one extra GEMV per layer)
  P2 prev-step   : actual ids of layer L+1 at the previous token (free)
  ORACLE         : actual ids of layer L+1 (ceiling; not implementable)

Baseline exposed bytes = recorded miss_bytes (ground truth; already reflects
the production LRU + pool caps). The prefetch arm is a stateful replay of the
same recorded semantics (per-pool access counter, victims = (usage, slot)
ascending); its fidelity is validated by running it with prefetch disabled and
comparing miss counts against the recorded ones.

GO gate (design page): exposed miss bytes drop >= 50% with k' <= 14.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

TOPK = 10


def load_capture(trace_dir: Path):
    rows = [
        json.loads(line) for line in (trace_dir / "trace.jsonl").read_text().splitlines() if line
    ]
    snapshot = json.loads((trace_dir / "snapshot.json").read_text())
    decode = [r for r in rows if r.get("kind") == "decode" and r.get("before_ids") is not None]
    hidden_list = torch.load(trace_dir / "hidden.pt", weights_only=False)
    hmap: dict[tuple[int, int], torch.Tensor] = {
        (step, layer): vec for step, layer, vec in hidden_list
    }
    return rows, snapshot, decode, hmap


def load_gates(model_path: str, n_layers: int) -> torch.Tensor:
    from freetoken.models.gguf.reader import _gguf_module, find_gguf_tensor

    gguf = _gguf_module()
    gates = []
    for b in range(n_layers):
        found = find_gguf_tensor(model_path, f"blk.{b}.ffn_gate_inp.weight")
        if found is None:
            raise SystemExit(f"router weight blk.{b}.ffn_gate_inp.weight not found")
        _path, t = found
        ne = [int(s) for s in t.shape]  # ggml order, fastest first: [in, out]
        block, _ts = gguf.GGML_QUANT_SIZES[t.tensor_type]
        assert block == 1, f"gate {b} quantized ({t.tensor_type}); expected BF16/F32"
        data = t.data
        if not data.flags.c_contiguous:
            data = np.ascontiguousarray(data)
        flat = data.reshape(-1)
        tt = int(t.tensor_type)
        if tt == 30:  # BF16
            w = torch.from_numpy(np.ascontiguousarray(flat.view(np.uint8))).view(torch.bfloat16)
        elif tt == 0:  # F32
            w = torch.from_numpy(np.ascontiguousarray(flat.astype(np.float32)))
        elif tt == 1:  # F16
            w = torch.from_numpy(np.ascontiguousarray(flat.view(np.uint8))).view(torch.float16)
        else:
            raise SystemExit(f"gate {b}: unsupported type {t.tensor_type}")
        gates.append(w.reshape(ne[1], ne[0]).float())  # [out=experts, in=hidden]
    return torch.stack(gates)


class PoolSim:
    """Stateful LRU replay using the recorded system semantics (matches
    benchmarks/replay_lru_trace.py actual_replay: usage stamped with a per-pool
    access counter = max(usage)+1; victims = (usage, slot) ascending)."""

    def __init__(self):
        self.pools: dict[int, tuple[list[int], list[int]]] = {}

    def sync(self, pool: int, ids: list[int], usage: list[int]) -> None:
        # The system's usage counter is a packed value (observed up to 2**40);
        # only its ORDER matters for LRU, so rank-compress on sync and keep
        # stamping max+1 afterwards (order-isomorphic evolution).
        order = sorted(range(len(usage)), key=lambda i: (usage[i], i))
        rank = [0] * len(usage)
        for r, slot in enumerate(order):
            rank[slot] = r + 1
        self.pools[pool] = (list(ids), rank)

    def _need(self, pool: int, cap_ids: list[int], cap_usage: list[int]) -> None:
        cur = self.pools.get(pool)
        if cur is None or len(cur[0]) != len(cap_ids):
            self.sync(pool, cap_ids, cap_usage)

    def ensure(self, pool: int, requested: list[int], before_ids, before_usage) -> list[int]:
        self._need(pool, before_ids, before_usage)
        ids, usage = self.pools[pool]
        step = max(usage, default=0) + 1
        idset = set(ids)
        for e in requested:
            if e in idset:
                usage[ids.index(e)] = step
        missing = [e for e in requested if e not in idset]
        if missing:
            victims = sorted(range(len(ids)), key=lambda i: (usage[i], i))[: len(missing)]
            for e, slot in zip(missing, victims):
                ids[slot] = e
                usage[slot] = step
        return missing

    def install_prefetch(self, pool: int, predicted: list[int], before_ids, before_usage) -> int:
        self._need(pool, before_ids, before_usage)
        ids, usage = self.pools[pool]
        step = max(usage, default=0) + 1
        idset = set(ids)
        todo = [e for e in predicted if e not in idset]
        if todo:
            victims = sorted(range(len(ids)), key=lambda i: (usage[i], i))[: len(todo)]
            for e, slot in zip(todo, victims):
                ids[slot] = e
                usage[slot] = step
        return len(todo)


def requested_of(row: dict) -> list[int]:
    return list(dict.fromkeys(row["requested_global_ids"]))


def row_bytes_of(row: dict) -> float:
    return (
        row["miss_bytes"] / row["missing"]
        if row["missing"]
        else float(sum(row["bank_bytes"].values()))
    )


def run_sim(decode, predictor, kprime, bytes_per_row, window_bytes, fidelity=False):
    """Prefetch-arm stateful replay. predictor(i) -> global ids to prefetch at
    row i for row i+1 (None = skip). fidelity=True disables prefetch and checks
    miss counts against the recorded ones."""
    sim = PoolSim()
    n = len(decode)
    st = dict(
        exposed=0.0,
        misses=0,
        pf_installed=0,
        pf_bytes=0.0,
        useful_bytes=0.0,
        wasted_bytes=0.0,
        recall_sum=0.0,
        recall_n=0,
        miss_recall_sum=0.0,
        miss_recall_n=0,
        precision_sum=0.0,
        mismatches=0,
        demand_bytes=0.0,
        window_over_rows=0,
        window_over_bytes=0.0,
    )
    pending: set[int] = set()
    for i, row in enumerate(decode):
        pool = row["pool"]
        req = requested_of(row)
        rb = bytes_per_row[i]
        missing = sim.ensure(pool, req, row["before_ids"], row["before_usage"])
        if fidelity:
            st["mismatches"] += len(missing) != row["missing"]
        st["exposed"] += len(missing) * rb
        st["misses"] += len(missing)
        if pending:
            used = pending & set(req)
            st["useful_bytes"] += len(used) * rb
            st["wasted_bytes"] += len(pending - used) * rb
            pending = set()
        pf_bytes = 0.0
        if predictor is not None and not fidelity and i + 1 < n:
            pred = predictor(i)
            if pred is not None:
                pred = list(dict.fromkeys(pred))[:kprime]
                nxt = decode[i + 1]
                nreq = set(requested_of(nxt))
                st["recall_sum"] += len(nreq & set(pred)) / min(TOPK, len(nreq))
                st["recall_n"] += 1
                st["precision_sum"] += len(nreq & set(pred)) / max(1, len(pred))
                base_miss_next = set(nreq) - set(nxt["before_ids"])
                if base_miss_next:
                    st["miss_recall_sum"] += len(base_miss_next & set(pred)) / len(base_miss_next)
                    st["miss_recall_n"] += 1
                n_inst = sim.install_prefetch(
                    nxt["pool"], pred, nxt["before_ids"], nxt["before_usage"]
                )
                st["pf_installed"] += n_inst
                pf_bytes = n_inst * bytes_per_row[i + 1]
                st["pf_bytes"] += pf_bytes
                ids_now = set(sim.pools[nxt["pool"]][0])
                pending = {g for g in pred if g in ids_now}
        demand = len(missing) * rb + pf_bytes
        st["demand_bytes"] += demand
        if demand > window_bytes:
            st["window_over_rows"] += 1
            st["window_over_bytes"] += demand - window_bytes
    st["n_rows"] = n
    return st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", required=True, help="capture dir (trace.jsonl + hidden.pt)")
    ap.add_argument("--model", required=True, help="GGUF checkpoint path (dir or shard)")
    ap.add_argument("--out", default=None, help="report dir (default <trace>/../pregate-report)")
    ap.add_argument("--itl", type=float, default=16.28, help="canonical ITL ms/token (61.44 TG)")
    ap.add_argument("--bw", type=float, default=53.5, help="measured gather GB/s")
    ap.add_argument("--exposed-ms", type=float, default=3.4, help="today's exposed gather ms/token")
    ap.add_argument(
        "--p1-cost-us", type=float, default=8.0, help="per-layer P1 GEMV+topk+launch us"
    )
    args = ap.parse_args()

    trace_dir = Path(args.trace)
    out_dir = Path(args.out or trace_dir.parent / "pregate-report")
    out_dir.mkdir(parents=True, exist_ok=True)

    _rows, snapshot, decode, hmap = load_capture(trace_dir)
    n_layers = snapshot["num_layers"]
    num_experts = snapshot["num_experts"]
    # hidden.pt is flushed periodically; a SIGKILLed server loses the tail.
    # Trim to the contiguous all-hidden prefix, rounded down to whole steps.
    keep = 0
    while keep < len(decode) and (decode[keep]["access_step"], decode[keep]["layer"]) in hmap:
        keep += 1
    n_hidden_rows = keep
    decode = decode[: keep - keep % n_layers]
    n_pools = len(set(snapshot["pool_of_layer"]))
    print(
        f"decode rows={len(decode)} (hidden present for {n_hidden_rows}) "
        f"moe_layers={n_layers} experts={num_experts} pools={n_pools}"
    )
    assert len(decode) % n_layers == 0, "decode rows not a multiple of MoE layers"
    assert len(decode) >= 50 * n_layers, f"only {len(decode) // n_layers} usable steps"
    n_steps = len(decode) // n_layers
    miss_h = [r["access_step"] for r in decode if (r["access_step"], r["layer"]) not in hmap]
    assert not miss_h, f"hidden missing for {len(miss_h)} rows, e.g. {miss_h[:5]}"

    H = torch.stack([hmap[(r["access_step"], r["layer"])] for r in decode]).float()
    bytes_per_row = [row_bytes_of(r) for r in decode]
    layer_of = [r["layer"] for r in decode]

    # window budget for the contention overlay
    window_bytes = (args.itl / n_layers) * 1e-3 * args.bw * 1e9
    print(f"per-layer copy window: {window_bytes / 1e6:.1f} MB (ITL {args.itl} ms / {n_layers})")

    gates = load_gates(args.model, n_layers)
    print(f"gates {tuple(gates.shape)}")

    # ---- sanity: same-layer early router must reproduce the recorded routing
    # (topk returns LOCAL expert ids; recorded expert_ids are local too)
    same = torch.zeros(len(decode), TOPK, dtype=torch.long)
    CH = 512
    for s in range(0, len(decode), CH):
        e = min(s + CH, len(decode))
        W = gates[layer_of[s:e]]
        logits = torch.einsum("ceh,ch->ce", W, H[s:e])
        same[s:e] = logits.topk(TOPK, dim=1).indices
    actual_local = [set(dict.fromkeys(r["expert_ids"])) for r in decode]
    sane = float(
        np.mean([len(actual_local[i] & set(same[i].tolist())) / TOPK for i in range(len(decode))])
    )
    print(f"sanity same-layer topk recall vs recorded routing: {sane:.4f} (must be ~1.0)")
    assert sane > 0.90, "router weight extraction/orientation broken; abort"

    # ---- fidelity of the stateful LRU replay (prefetch disabled)
    fid = run_sim(decode, None, 0, bytes_per_row, window_bytes, fidelity=True)
    exposed_base = sum(r["miss_bytes"] for r in decode)
    print(
        f"LRU fidelity: {fid['mismatches']}/{fid['n_rows']} mismatched rows; "
        f"sim exposed {fid['exposed'] / 1e6:.1f} MB vs recorded {exposed_base / 1e6:.1f} MB"
    )

    cos = torch.nn.functional.cosine_similarity(H[:-1], H[1:], dim=1)
    print(f"cos(h_L, h_L+1): mean={cos.mean():.4f} p10={cos.quantile(0.10):.4f}")

    # ---- predictors (all must emit GLOBAL ids in the TARGET row's layer space)
    scores_cache: dict[int, list[int]] = {}

    def target_layer(i: int) -> int:
        return layer_of[i + 1]

    def early_ids(i: int) -> list[int]:
        v = scores_cache.get(i)
        if v is None:
            nl = target_layer(i)
            top = (gates[nl] @ H[i]).topk(14).indices.tolist()
            v = [nl * num_experts + e for e in top]
            if len(scores_cache) > 8192:
                scores_cache.clear()
            scores_cache[i] = v
        return v

    def p0(i):
        # persistence: layer i's LOCAL expert ids re-expressed for the target layer
        nl = target_layer(i)
        return [nl * num_experts + e for e in decode[i]["expert_ids"]]

    def p2(i):
        # previous token step, same layer as the target row
        j = i + 1 - n_layers
        return requested_of(decode[j]) if j >= 0 else None

    def p1p2(i):
        # interleave early-router top picks with prev-step ids, dedup
        a = early_ids(i)
        b = p2(i) or []
        out = []
        for x, y in zip(a, b):
            out.append(x)
            out.append(y)
        out.extend(a[len(b) :])
        return list(dict.fromkeys(out))

    def oracle(i):
        return requested_of(decode[i + 1])

    variants = [
        ("P0-persistence", p0, TOPK),
        ("P2-prev-step", p2, TOPK),
        ("P1-early-k10", early_ids, 10),
        ("P1-early-k12", early_ids, 12),
        ("P1-early-k14", early_ids, 14),
        ("P1P2-union-k14", p1p2, 14),
        ("ORACLE-k10", oracle, TOPK),
    ]

    mb = 1e6
    results = {}
    base_mb_tok = exposed_base / mb / n_steps
    fid_exposed = fid["exposed"]
    for name, pred, kp in variants:
        st = run_sim(decode, pred, kp, bytes_per_row, window_bytes)
        drop = 1.0 - st["exposed"] / max(exposed_base, 1.0)
        # sim-vs-sim drop: both arms share the same LRU model bias (fidelity gap
        # vs the recorded system), so this is the unbiased effect of prefetching
        drop_sim = 1.0 - st["exposed"] / max(fid_exposed, 1.0)
        cost_ms = n_layers * args.p1_cost_us / 1000.0 if name.startswith("P1") else 0.0
        saved_ms = args.exposed_ms * drop_sim
        # contention: bytes over the per-layer window add serial copy time
        over_ms = st["window_over_bytes"] / (args.bw * 1e9) * 1000.0 / n_steps
        net_ms = saved_ms - cost_ms - over_ms
        new_itl = args.itl - net_ms
        tg = 1000.0 / new_itl if new_itl > 0 else float("inf")
        results[name] = dict(
            k=kp,
            recall=st["recall_sum"] / max(st["recall_n"], 1),
            precision=st["precision_sum"] / max(st["recall_n"], 1),
            miss_recall=st["miss_recall_sum"] / max(st["miss_recall_n"], 1),
            exposed_base_mb_token=base_mb_tok,
            exposed_pre_mb_token=st["exposed"] / mb / n_steps,
            drop=drop,
            drop_sim=drop_sim,
            useful_mb_token=st["useful_bytes"] / mb / n_steps,
            wasted_mb_token=st["wasted_bytes"] / mb / n_steps,
            prefetch_mb_token=st["pf_bytes"] / mb / n_steps,
            extra_misses_token=(st["misses"] - fid["misses"]) / n_steps,
            predictor_cost_ms=cost_ms,
            contention_over_ms_token=over_ms,
            window_over_rows=st["window_over_rows"],
            saved_ms=saved_ms,
            net_ms=net_ms,
            projected_tg=tg,
        )
        r = results[name]
        print(
            f"{name:16s} k={kp:2d} recall={r['recall']:.3f} miss_rec={r['miss_recall']:.3f} "
            f"drop_sim={r['drop_sim']:+.1%} (rec {r['drop']:+.1%}) "
            f"useful={r['useful_mb_token']:6.1f}MB "
            f"wasted={r['wasted_mb_token']:6.1f}MB extra_miss={r['extra_misses_token']:+5.1f}/tok "
            f"cont={r['contention_over_ms_token']:.2f}ms net={r['net_ms']:+.2f}ms TG~{tg:.1f}"
        )

    go = any(
        v["drop_sim"] >= 0.5 and v["k"] <= 14 and v["net_ms"] >= 0.3
        for k, v in results.items()
        if not k.startswith("ORACLE")
    )
    verdict = "GO" if go else "NO-GO"
    print(f"\nGATE (drop>=50% with k'<=14, net>=+0.3ms): {verdict}")
    (out_dir / "pregate_report.json").write_text(
        json.dumps(
            dict(
                verdict=verdict,
                sanity_same_layer_recall=sane,
                lru_fidelity_mismatches=fid["mismatches"],
                cos_h_h_next_mean=float(cos.mean()),
                cos_h_h_next_p10=float(cos.quantile(0.10)),
                n_steps=n_steps,
                n_layers=n_layers,
                num_experts=num_experts,
                itl=args.itl,
                bw=args.bw,
                exposed_ms=args.exposed_ms,
                p1_cost_us=args.p1_cost_us,
                window_bytes=window_bytes,
                results=results,
            ),
            indent=1,
        )
    )
    print(f"report: {out_dir / 'pregate_report.json'}")


if __name__ == "__main__":
    main()
