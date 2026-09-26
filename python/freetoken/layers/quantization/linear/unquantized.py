"""bf16 Linear: one kernel (torch), no scheme."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import LinearKernel, LinearMethod


def small_batch_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
    """``F.linear`` for a 2-D activation; 2..8 rows (MTP verify/draft windows) go through
    ``w @ x.T``, which cuBLAS runs as a batched GEMV: measured on sm_120, [48, 5120] bf16
    at 2-4 rows 34 us -> 10 us (x @ w.T picks a slow tiled GEMM there)."""
    if x.dim() == 2 and 2 <= x.shape[0] <= 8:
        out = (w @ x.T).T.contiguous()  # callers (e.g. hc_silu) require row-major
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
