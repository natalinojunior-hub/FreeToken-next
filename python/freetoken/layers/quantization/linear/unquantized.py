"""bf16 Linear: one kernel (torch), no scheme."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F

from ....kernel.triton.small_m_gemm import small_m_linear
from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import LinearKernel, LinearMethod

# Kill switch for the small-M dispatch below. Set FREETOKEN_SMALL_M_SPLITK=0 to restore the
# legacy ``w @ x.T`` path for a one-flag A/B.
_SMALL_M_DISPATCH = os.environ.get("FREETOKEN_SMALL_M_SPLITK", "1") != "0"
# Below ~2 cuBLAS N-tiles (128 each) cuBLAS's ``x @ w.T`` mis-tiles small-N and runs 10-20 us;
# the split-K GEMV creates the missing parallelism there instead. A hardware constant, not a
# tuned knob. At/above it, cuBLAS ``F.linear`` beats both ``w @ x.T`` and split-K.
_SMALL_M_SPLITK_MAX_N = 256


def small_batch_linear(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None
) -> torch.Tensor:
    """``F.linear`` for a 2-D activation; the 2..8-row band (MTP verify windows) is dispatched
    per shape instead of through cuBLAS's tiled path.

    The shipped ``w @ x.T`` (measured on sm_120, [48, 5120] bf16, 2-4 rows 34 us -> 10 us) is
    pathological at the M=3 MTP verify size for every resident bf16 shape -- the decode census
    shows it as the ``cutlass_75_*`` family at 24 us on the router gate, 6x off the DRAM ceiling.
    Graph-replay microbench at M=3 picks the winner by N:

    * ``N >= 256`` (router gate [512,2560], hc_up [10240,320], hc_down+inject [336,10240],
      ple_key [10240,2560]): ``F.linear`` -- 2-6x faster than ``w @ x.T``, and faster than
      split-K (cuBLAS tiles a wide N well; split-K's extra reduce loses).
    * ``N < 256`` (ssm [48,2560], indexer k [128,2560]): ``small_m_linear`` split-K GEMV --
      4-5x faster than ``w @ x.T``, and cuBLAS ``F.linear`` mis-tiles small-N here.

    Only the 2..8 band changes; M=1 (raw decode and MTP drafts) stays on ``F.linear`` exactly
    as before -- widening downward was measured at -19.3% (see campaign37 m1-regression note).
    """
    if x.dim() == 2 and 2 <= x.shape[0] <= 8:
        if os.environ.get("FREETOKEN_ROW_INVARIANT_LINEAR") == "1":
            return torch.cat([F.linear(row.unsqueeze(0), w, b) for row in x])
        if _SMALL_M_DISPATCH:
            if w.shape[0] < _SMALL_M_SPLITK_MAX_N:
                return small_m_linear(x, w, b)  # split-K; internal F.linear fallback off-envelope
            return F.linear(x, w, b)
        out = (w @ x.T).T.contiguous()  # legacy: callers (e.g. hc_silu) require row-major
        return out + b if b is not None else out
    return F.linear(x, w, b)


class TorchLinearKernel(LinearKernel):
    name = "torch"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        # an fp32 activation stream (DeepSeek-V4's compressors) upcasts the bf16 weight on the fly, as the reference does
        w, b = layer.weight, layer.bias
        if w.dtype != x.dtype:
            w = w.to(x.dtype)
            b = b.to(x.dtype) if b is not None else None
        return small_batch_linear(x, w, b)


@register_method(QuantKind.NONE, LayerKind.LINEAR)
class UnquantizedLinearMethod(LinearMethod):
    candidates = (TorchLinearKernel,)

    def create_weights(self, layer: Any) -> None:
        g = self.cfg
        layer.weight = torch.empty(g.out_features, g.in_features)
