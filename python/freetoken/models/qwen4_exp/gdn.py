from __future__ import annotations

import torch

from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet


class Qwen4ExpGatedDeltaNet(Qwen3_5GatedDeltaNet):
    """Qwen3.5's GatedDeltaNet (prefill, decode, fused multi-row MTP verify) with the gate
    activation from ``LinearGatedDeltaGroupConfig`` ("sigmoid" for Qwen3.8-Flash-Next) and an
    optional input gather in front of ``out_proj``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set by a loader whose out_proj columns stay in a different head order: the input is
        # gathered through it first (see qwen4_exp.gguf._tiled_input_perm). None = identity.
        self.out_proj_in_perm: torch.Tensor | None = None

    def _out_proj_input(self, out: torch.Tensor) -> torch.Tensor:
        if self.out_proj_in_perm is not None:
            out = out.index_select(-1, self.out_proj_in_perm)
        return out


__all__ = ["Qwen4ExpGatedDeltaNet"]
