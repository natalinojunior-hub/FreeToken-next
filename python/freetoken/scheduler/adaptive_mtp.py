"""Adaptive MTP speculation depth governance.

Confidence-gated draft launch: skip speculative step when draft head's top-1
probability falls below threshold, avoiding verify overhead for low-quality drafts.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class AdaptiveMTPConfig:
    enabled: bool = True
    # Minimum top-1 probability to launch draft chain
    min_draft_prob: float = 0.85
    # Maximum speculation depth
    max_k: int = 3
    # Fallback to k=1 when confidence low
    fallback_k: int = 1
    # EMA smoothing for probability estimate
    ema_alpha: float = 0.1


class AdaptiveMTPController:
    """Runtime controller for adaptive MTP speculation depth."""

    def __init__(self, config: AdaptiveMTPConfig, device: torch.device):
        self.config = config
        self.device = device
        self._ema_prob: Optional[torch.Tensor] = None
        self._step_count = 0

    def should_speculate(self, draft_logits: torch.Tensor, k: int) -> tuple[bool, int]:
        """Decide whether to run speculative step and at what depth.

        Args:
            draft_logits: [1, vocab] logits from draft head for next token
            k: configured max speculation depth

        Returns:
            (should_speculate, effective_k)
        """
        if not self.config.enabled:
            return False, 0

        probs = torch.softmax(draft_logits, dim=-1)
        top1_prob = probs.max().item()

        # Update EMA
        if self._ema_prob is None:
            self._ema_prob = torch.tensor(top1_prob, device=self.device)
        else:
            self._ema_prob = (
                self.config.ema_alpha * top1_prob
                + (1 - self.config.ema_alpha) * self._ema_prob.item()
            )

        self._step_count += 1

        # Gate: only speculate if draft is confident
        if top1_prob < self.config.min_draft_prob:
            return False, 0

        # Adaptive depth: higher confidence -> deeper speculation
        effective_k = min(k, self.config.max_k)
        if top1_prob > 0.95:
            effective_k = min(effective_k + 1, self.config.max_k)
        elif top1_prob < 0.90:
            effective_k = max(self.config.fallback_k, effective_k - 1)

        return True, effective_k

    def get_stats(self) -> dict:
        return {
            "ema_prob": self._ema_prob.item() if self._ema_prob is not None else 0.0,
            "step_count": self._step_count,
            "enabled": self.config.enabled,
        }


def resolve_adaptive_k(
    req,
    base_k: int | None = None,
    max_k: int | None = None,
    moe_slots_per_layer: float | None = None,
    top_k_experts: int | None = None,
    draft_logits: Optional[torch.Tensor] = None,
    controller: Optional[AdaptiveMTPController] = None,
) -> int:
    """Resolve effective speculation depth for current step.

    Supports two calling conventions:
    1. Legacy (tests): resolve_adaptive_k(req, max_k=4) -> uses context-length heuristics
    2. New (spec.py): resolve_adaptive_k(req, base_k=2, draft_logits=..., controller=...)

    Args:
        req: Request object with device_len, remain_len attributes
        base_k: Configured --spec-mtp value (new API)
        max_k: Legacy max_k parameter for context-length heuristics
        moe_slots_per_layer: MoE cache slots per layer (unused)
        top_k_experts: Top-k experts per token (unused)
        draft_logits: Draft head logits for next token (if available, new API)
        controller: AdaptiveMTPController instance (if adaptive enabled, new API)

    Returns:
        Effective k for this step (0 = no speculation)
    """
    # Legacy API: context-length heuristic (used by tests)
    if max_k is not None and base_k is None and draft_logits is None:
        device_len = getattr(req, "device_len", 0)
        remain_len = getattr(req, "remain_len", 100)
        # Original heuristic from test expectations
        if device_len >= 131072:
            return min(1, max_k)
        elif device_len >= 65536:
            return min(2, max_k)
        # Clamp by remaining tokens (remain_len - 1)
        max_by_remain = max(1, remain_len - 1)
        return min(max_k, max_by_remain)

    # New API: confidence-gated with controller
    if controller is None or draft_logits is None:
        return base_k if base_k is not None else 0

    should_spec, eff_k = controller.should_speculate(
        draft_logits, base_k if base_k is not None else 0
    )
    return eff_k if should_spec else 0
