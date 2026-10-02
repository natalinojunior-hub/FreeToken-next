"""bf16 Linear: one kernel (torch), no scheme."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F

from ....core import row_invariant_rows
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


_UNIT_SCALES: dict[tuple[int, torch.device], torch.Tensor] = {}


def _bf16_gemv_rows(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    from ....kernel.triton.fp8_pertensor_linear import _gemv_rows

    key = (w.shape[0], w.device)
    ones = _UNIT_SCALES.get(key)
    if ones is None:
        ones = _UNIT_SCALES[key] = torch.ones(w.shape[0], dtype=torch.float32, device=w.device)
    return _gemv_rows(x, w, ones, x.dtype)


def small_batch_linear(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None
) -> torch.Tensor:
    """Small bf16 decode/verify rows use the same per-weight reduction policy.

    Graph replay favors cuBLAS at M=1 for 64/256-output gates (1.6/1.9 us versus
    split-K's 1.9/2.2 us). Scalar gates and wider projections favor shared GEMV.
    Verify windows use that same choice, keeping each row bitwise equal to M=1.
    """
    if (
        x.dim() == 2
        and (x.shape[0] == 1 or row_invariant_rows(x.shape[0]))
        and x.is_cuda
        and x.dtype == w.dtype == torch.bfloat16
        and (w.shape[0] == 1 or w.shape[0] > 256)
    ):
        # decode rows and MTP verify windows share one split-K GEMV: each verify row is
        # bitwise its M==1 decode row while the weight tile is read once for the window
        out = _bf16_gemv_rows(x, w)
        return out + b if b is not None else out
    if x.dim() == 2 and 2 <= x.shape[0] <= 8:
        if row_invariant_rows(x.shape[0]):
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
