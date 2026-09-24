"""Grouped expert GEMM over native GGUF banks (borrowed ggml MoE kernels).

Ports vLLM/sglang's ``_fused_moe_gguf`` MMVQ path onto FreeToken's offload-cache
interface: the experts are streamed to the GPU as packed block bytes and
dequantized *inside* ``ggml_moe_a8_vec`` -- no bf16 expert copy is materialized. We
use the MMVQ (vector) kernel for both prefill and decode: it consumes ``topk_ids``
directly (no ``moe_align_block_size`` needed) and on small batches it is the right
choice anyway. ``topk_ids`` already index the streamed cache slots (decode) or the
materialized layer positions (prefill).

This module is general over any quantization type supported by the ``ggml_moe_a8_vec``
kernel (all types in ``MOE_VEC_TYPES``, which includes all 19 quantized types). Q4_0
is currently the only type the rest of the pipeline plumbs through; support for other
types is added by parametrizing the quant type at the MoE bank loader, dequant.py, and
moe/expert_banks.py level.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import time

import torch

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from freetoken.models.gguf.dequant import GGML_Q4_0, MOE_VEC_TYPES

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}


class _MoeEvents:
    """Opt-in CUDA event collector; one synchronization occurs at process exit."""

    def __init__(self) -> None:
        self.path = os.environ.get("FREETOKEN_MOE_EVENTS")
        self.limit = int(os.environ.get("FREETOKEN_MOE_EVENTS_MAX", "4096"))
        self.items: list[tuple[str, object, object]] = []
        self.anchor = None
        self.record_ns = 0

    def record(self, label: str, fn: object) -> object:
        if not self.path or len(self.items) >= self.limit or not torch.cuda.is_available():
            return fn()
        t0 = time.perf_counter_ns()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if self.anchor is None:
            self.anchor = torch.cuda.Event(enable_timing=True)
            self.anchor.record()
        start.record()
        value = fn()
        end.record()
        self.record_ns += time.perf_counter_ns() - t0
        self.items.append((label, start, end))
        return value

    def flush(self) -> None:
        if not self.path or not self.items:
            return
        try:
            torch.cuda.synchronize()
            intervals = [
                {"label": label, "start_ms": self.anchor.elapsed_time(start), "end_ms": self.anchor.elapsed_time(end)}
                for label, start, end in self.items
            ]
            with open(self.path, "w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "event_count": len(intervals),
                        "record_overhead_ms": self.record_ns / 1e6,
                        "intervals": intervals,
                    },
                    stream,
                )
        except Exception:
            return


_MOE_EVENTS = _MoeEvents()
atexit.register(_MOE_EVENTS.flush)


def _flush_moe_events_on_term(signum: int, _frame: object) -> None:
    _MOE_EVENTS.flush()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


if _MOE_EVENTS.path:
    signal.signal(signal.SIGTERM, _flush_moe_events_on_term)


# From this many tokens (measured crossover, 512 experts x top-10) the GEMV kernel, which re-reads each expert's weights per routed
# row, loses to dequantizing the used experts and running the bf16 grouped GEMM.
DEQUANT_MIN_TOKENS = 512
# Experts dequantized at once: bounds the bf16 transient (~10 MB per expert here), which
# the memory planner otherwise takes from the expert cache.
DEQUANT_EXPERT_CHUNK = 16


def _gather_experts(
    bank: torch.Tensor, experts: torch.Tensor, first: int, last: int
) -> torch.Tensor:
    """``bank[experts]`` for a packed uint8 bank: a view when the ids are the contiguous run
    ``first..last`` (a prefill uses nearly every expert), else a gather in the widest element
    that divides an expert's bytes (a byte-wise index_select is ~8x slower than int64)."""
    flat = bank.reshape(bank.shape[0], -1)
    if last - first + 1 == experts.numel():
        return flat[first : last + 1]
    for dtype in (torch.int64, torch.int32, torch.int16):
        size = dtype.itemsize
        if flat.shape[1] % size == 0 and flat.data_ptr() % size == 0:
            return flat.view(dtype).index_select(0, experts).view(torch.uint8)
    return flat.index_select(0, experts)


def _fused_experts_dequant(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,
    down_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    quant_type: int,
    down_quant_type: int,
) -> torch.Tensor:
    """Large-batch GGUF MoE: dequantize only the routed experts, a chunk at a time, and run
    the bf16 fused-MoE kernel on the (token, expert) rows each chunk owns."""
    from freetoken.kernel.gguf import ggml_dequantize
    from freetoken.moe.fused import fused_experts_impl

    num_tokens, h = hidden_states.shape
    n2 = gate_up_q.shape[1]
    inter = n2 // 2
    top_k = topk_ids.shape[1]
    flat = topk_ids.reshape(-1).long()
    order = torch.argsort(flat)
    sorted_ids = flat[order]
    used, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    counts_host = counts.tolist()
    used_host = used.tolist()
    token_of = order // top_k
    weights = topk_weights.reshape(-1)[order].unsqueeze(1)
    # Each (token, slot) pair is written once, then summed over top_k in a fixed order:
    # index_add_ into bf16 rounded per expert in atomic, run-dependent order.
    out = hidden_states.new_empty(num_tokens * top_k, h)
    start = 0
    for c0 in range(0, used.numel(), DEQUANT_EXPERT_CHUNK):
        experts = used[c0 : c0 + DEQUANT_EXPERT_CHUNK]
        n = sum(counts_host[c0 : c0 + DEQUANT_EXPERT_CHUNK])
        rows = slice(start, start + n)
        start += n
        e = experts.numel()
        ends = used_host[c0], used_host[c0 + e - 1]
        w1 = ggml_dequantize(
            _gather_experts(gate_up_q, experts, *ends).reshape(e * n2, -1),
            quant_type,
            e * n2,
            h,
            hidden_states.dtype,
        ).view(e, n2, h)
        w2 = ggml_dequantize(
            _gather_experts(down_q, experts, *ends).reshape(e * h, -1),
            down_quant_type,
            e * h,
            inter,
            hidden_states.dtype,
        ).view(e, h, inter)
        local = torch.searchsorted(experts, sorted_ids[rows]).to(torch.int32).unsqueeze(1)
        x = hidden_states.index_select(0, token_of[rows])
        y = fused_experts_impl(x, w1, w2, weights[rows].contiguous(), local, activation)
        out[order[rows]] = y
    return out.view(num_tokens, top_k, h).sum(dim=1, dtype=torch.float32).to(hidden_states.dtype)


def fused_experts_gguf(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, H//32*18] uint8 (or other quant format)
    down_q: torch.Tensor,  # [num_slots, H, I//32*18] uint8 (or other quant format)
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    quant_type: int,
    down_quant_type: int | None = None,
) -> torch.Tensor:
    """Fused GGUF MoE expert compute over any MMVQ-supported quantization type.

    This kernel operates directly on packed quantized weights (no materialization to bf16);
    dequantization happens inside the ``ggml_moe_a8_vec`` CUDA kernel. ``quant_type`` must be
    in ``MOE_VEC_TYPES``, which mirrors the supported types in ``ggml_moe_a8_vec``
    (gguf_kernel.cu:559).

    ``quant_type`` is the gate_up bank's type; ``down_quant_type`` defaults to it. They may
    differ because gate_up and down are separate banks with separate slot pools, and
    llama.cpp routinely quantizes the down projection differently from gate/up. What may
    NOT differ is the type *within* one bank across layers -- that pool is one allocation.
    """
    from freetoken.kernel.gguf import ggml_moe_a8_vec

    if down_quant_type is None:
        down_quant_type = quant_type
    for label, qt in (("gate_up", quant_type), ("down", down_quant_type)):
        if qt not in MOE_VEC_TYPES:
            from freetoken.models.gguf.dequant import GGML_NAME

            raise NotImplementedError(
                f"fused GGUF MoE kernel does not support quant type "
                f"{GGML_NAME.get(qt, qt)} for the {label} bank "
                f"(only {sorted(MOE_VEC_TYPES)} supported)"
            )

    act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")

    num_tokens = hidden_states.shape[0]
    n2 = gate_up_q.shape[1]  # 2 * intermediate
    h = down_q.shape[1]  # hidden
    top_k = topk_ids.shape[1]
    qt = int(quant_type)
    if num_tokens >= DEQUANT_MIN_TOKENS and not torch.cuda.is_current_stream_capturing():
        return _fused_experts_dequant(
            hidden_states,
            gate_up_q,
            down_q,
            topk_weights,
            topk_ids,
            activation,
            qt,
            int(down_quant_type),
        )

    # gate_up: [num_tokens*top_k, 2I] -> activation -> [num_tokens*top_k, I]
    if _MOE_EVENTS.path:
        gate_up = _MOE_EVENTS.record(
            "gate_up", lambda: ggml_moe_a8_vec(hidden_states, gate_up_q, topk_ids, top_k, qt, n2, num_tokens)
        )
    else:
        gate_up = ggml_moe_a8_vec(hidden_states, gate_up_q, topk_ids, top_k, qt, n2, num_tokens)
    inter = act_fn(gate_up)
    # down: each of the num_tokens*top_k intermediate rows uses its own expert id.
    if _MOE_EVENTS.path:
        out = _MOE_EVENTS.record(
            "down", lambda: ggml_moe_a8_vec(inter, down_q, topk_ids, 1, int(down_quant_type), h, num_tokens * top_k)
        )
    else:
        out = ggml_moe_a8_vec(inter, down_q, topk_ids, 1, int(down_quant_type), h, num_tokens * top_k)
    out = out.reshape(num_tokens, top_k, h) * topk_weights.reshape(num_tokens, top_k, 1).to(
        out.dtype
    )
    return out.sum(dim=1)


def fused_experts_gguf_q4_0(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, H//32*18] uint8
    down_q: torch.Tensor,  # [num_slots, H, I//32*18] uint8
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    """GGUF Q4_0 MoE (backward-compat wrapper).

    This is a thin wrapper around ``fused_experts_gguf`` that hardcodes the Q4_0 type.
    All existing callers use this for now; the general function is available for future
    multi-quant pipelines.
    """
    return fused_experts_gguf(
        hidden_states, gate_up_q, down_q, topk_weights, topk_ids, activation, int(GGML_Q4_0)
    )


__all__ = ["fused_experts_gguf", "fused_experts_gguf_q4_0"]
