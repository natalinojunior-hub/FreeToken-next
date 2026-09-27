"""Opt-in tracing for eager MoE offload-cache calls."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


class MoeTracer:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._fh = (self.out_dir / "trace.jsonl").open("w")
        self._step = 0
        self.initial_written = False
        # Opt-in router-input capture (FREETOKEN_MOE_TRACE_HIDDEN=1): stores the
        # decode MoE input hidden row per (access_step, layer) for offline
        # cross-layer routing-prediction studies. Keyed by the access_step the
        # next record() will write; saved to hidden.pt on close().
        self.capture_hidden = os.getenv("FREETOKEN_MOE_TRACE_HIDDEN", "").strip() == "1"
        self._hidden: list[tuple[int, int, torch.Tensor]] = []
        if self.capture_hidden:
            import atexit

            atexit.register(self._save_hidden)

    def _save_hidden(self) -> None:
        if self._hidden:
            torch.save(self._hidden, self.out_dir / "hidden.pt")

    def record_hidden(self, layer_id: int, hidden_states: torch.Tensor, kind: str) -> None:
        if not self.capture_hidden or kind != "decode" or _capturing():
            return
        if hidden_states.shape[0] != 1:
            return
        self._hidden.append(
            (self._step, layer_id, hidden_states.reshape(-1).to(torch.bfloat16).cpu().clone())
        )
        if len(self._hidden) % 2048 == 0:
            self._save_hidden()

    @classmethod
    def from_env(cls) -> MoeTracer | None:
        path = os.getenv("FREETOKEN_MOE_TRACE", "").strip()
        return cls(Path(path)) if path else None

    def write_snapshot(self, cache: Any) -> None:
        layers = []
        for layer_id in range(cache.num_layers):
            per_bank = {
                name: cache.bank_sources[name][layer_id][0].numel()
                * cache.bank_sources[name][layer_id][0].element_size()
                for name in cache.bank_schema
            }
            bank_types = {
                name: cache._get_layer_quant_type(layer_id, name) for name in cache.bank_schema
            }
            layers.append(
                {
                    "layer": layer_id,
                    "bank_bytes": per_bank,
                    "bank_types": bank_types,
                    "total_bytes": sum(per_bank.values()),
                }
            )
        pools = []
        for pool_id, pool in enumerate(cache.pools):
            pools.append(
                {
                    "layers": list(pool.layers),
                    "capacity_rows": cache.pool_caps[pool_id],
                    "row_bytes": list(pool.row_bytes),
                    "capacity_bytes": cache.pool_caps[pool_id] * sum(pool.row_bytes),
                }
            )
        total_bytes = sum(pool["capacity_bytes"] for pool in pools)
        (self.out_dir / "snapshot.json").write_text(
            json.dumps(
                {
                    "num_layers": cache.num_layers,
                    "num_experts": cache.num_experts,
                    "quant_format": cache.quant_format,
                    "cache_size_rows": cache.cache_size,
                    "cache_capacity_bytes": total_bytes,
                    "bank_schema": list(cache.bank_schema),
                    "pool_of_layer": list(cache.pool_of_layer),
                    "pool_caps": list(cache.pool_caps),
                    "pools": pools,
                    "layers": layers,
                    "capture_limitations": (
                        "Python trace hooks do not run on CUDA graph replay; collection is "
                        "eager-only and trace-enabled graph capture is skipped."
                    ),
                }
            )
        )

    def write_initial_residency(self, ids: list[int], usage: list[int]) -> None:
        (self.out_dir / "initial_residency.json").write_text(
            json.dumps({"id_of_slot": ids, "usage": usage})
        )
        self.initial_written = True

    def record(
        self,
        *,
        kind: str,
        layer_id: int,
        pool_id: int,
        expert_ids: list[int],
        missing: int,
        miss_bytes: int,
        bank_bytes: dict[str, int],
        evicted_ids: list[int],
        resident_rows: int,
        transfer_ms: float | None,
        available_vram_bytes: int | None,
        requested_global_ids: list[int] | None = None,
        hit_global_ids: list[int] | None = None,
        miss_global_ids: list[int] | None = None,
        before_ids: list[int] | None = None,
        after_ids: list[int] | None = None,
        before_usage: list[int] | None = None,
        after_usage: list[int] | None = None,
        victim_slots: list[int] | None = None,
    ) -> None:
        if _capturing():
            return
        self._fh.write(
            json.dumps(
                {
                    "access_step": self._step,
                    "t": time.monotonic(),
                    "kind": kind,
                    "layer": layer_id,
                    "pool": pool_id,
                    "expert_ids": expert_ids,
                    "missing": missing,
                    "miss_bytes": miss_bytes,
                    "bank_bytes": bank_bytes,
                    "evicted_ids": evicted_ids,
                    "resident_rows": resident_rows,
                    "transfer_ms": transfer_ms,
                    "available_vram_bytes": available_vram_bytes,
                    "requested_global_ids": requested_global_ids,
                    "hit_global_ids": hit_global_ids,
                    "miss_global_ids": miss_global_ids,
                    "before_ids": before_ids,
                    "after_ids": after_ids,
                    "before_usage": before_usage,
                    "after_usage": after_usage,
                    "victim_slots": victim_slots,
                }
            )
            + "\n"
        )
        self._fh.flush()
        self._step += 1

    def close(self) -> None:
        if self._hidden:
            torch.save(self._hidden, self.out_dir / "hidden.pt")
            self._hidden.clear()
        self._fh.close()
