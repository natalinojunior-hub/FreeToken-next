"""TCQ / VBR (Variable Bitrate Context Quantization) policy for TurboKV.

Plans per-(layer, side) tier schedules to protect high-variance attention layers
(e.g., K-side logit peaks and sensitive early/late layers) with turbo4 while
compressing more tolerant layers/sides with turbo3 to optimize bandwidth and TG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Tier = Literal["turbo3", "turbo4", "bf16"]


@dataclass(frozen=True)
class TierSchedule:
    """Per-(layer, side) precision schedule across attention layers."""

    # key_tiers: list of Tier per attention layer
    key_tiers: tuple[Tier, ...]
    # value_tiers: list of Tier per attention layer
    value_tiers: tuple[Tier, ...]

    def get_tier(self, layer_idx: int, side: Literal["k", "v"]) -> Tier:
        tiers = self.key_tiers if side == "k" else self.value_tiers
        if layer_idx < 0 or layer_idx >= len(tiers):
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {len(tiers)})")
        return tiers[layer_idx]

    @property
    def num_layers(self) -> int:
        return len(self.key_tiers)

    @property
    def has_mixed_tiers(self) -> bool:
        all_tiers = set(self.key_tiers) | set(self.value_tiers)
        return len(all_tiers) > 1


def plan_vbr_schedule(
    num_layers: int,
    base_format: str = "turbo4",
    vbr_policy: str = "balanced",
) -> TierSchedule:
    """Build a deterministic per-(layer, side) TCQ/VBR schedule.

    Policies:
    - 'uniform': All layers/sides use base_format.
    - 'balanced':
        * Value-side error averages across attention weights (NMSE floor), so V can
          safely use turbo3 on intermediate layers.
        * Key-side enters exp(), so K-side on early layers (anchor tokens) and late
          layers stays turbo4, while middle layers can transition to turbo3.
    - 'aggressive':
        * All V-side uses turbo3.
        * K-side uses turbo3 on all except first and last 20% of layers.
    """
    if base_format not in ("turbo3", "turbo4", "vbr"):
        # Default uniform
        return TierSchedule(
            key_tiers=tuple("turbo4" for _ in range(num_layers)),
            value_tiers=tuple("turbo4" for _ in range(num_layers)),
        )

    if vbr_policy == "uniform" or base_format == "turbo3":
        fmt = "turbo3" if base_format == "turbo3" else "turbo4"
        return TierSchedule(
            key_tiers=tuple(fmt for _ in range(num_layers)),
            value_tiers=tuple(fmt for _ in range(num_layers)),
        )

    k_tiers: list[Tier] = []
    v_tiers: list[Tier] = []

    for i in range(num_layers):
        rel_pos = i / max(1, num_layers - 1)
        is_anchor = (rel_pos < 0.2) or (rel_pos > 0.8)

        if vbr_policy == "aggressive":
            # Aggressive: V is always turbo3, K is turbo4 only for anchors
            k_tiers.append("turbo4" if is_anchor else "turbo3")
            v_tiers.append("turbo3")
        else:
            # Balanced: K is turbo4 everywhere or anchors; V is turbo4 on anchors, turbo3 in middle
            k_tiers.append("turbo4")
            v_tiers.append("turbo4" if is_anchor else "turbo3")

    return TierSchedule(key_tiers=tuple(k_tiers), value_tiers=tuple(v_tiers))


class TCQPolicy:
    """Manages TCQ / VBR (Variable Bitrate Context Quantization) policies and schedules."""

    def __init__(
        self,
        num_layers: int,
        base_format: str = "turbo4",
        vbr_policy: str = "balanced",
    ) -> None:
        self.num_layers = num_layers
        self.base_format = base_format
        self.vbr_policy = vbr_policy
        self.schedule = plan_vbr_schedule(num_layers, base_format, vbr_policy)

    def get_tier(self, layer_idx: int, side: Literal["k", "v"]) -> Tier:
        return self.schedule.get_tier(layer_idx, side)

    @property
    def has_mixed_tiers(self) -> bool:
        return self.schedule.has_mixed_tiers
