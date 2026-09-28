"""MTP speculation depth per step (the configured ``--spec-mtp`` k in serving)."""

from __future__ import annotations

from collections import deque
from math import inf, isfinite, sqrt


def resolve_adaptive_k(
    req,
    base_k: int | None = None,
    max_k: int | None = None,
    moe_slots_per_layer: float | None = None,
    top_k_experts: int | None = None,
) -> int:
    """Resolve effective speculation depth for current step.

    Supports two calling conventions:
    1. Legacy (tests): resolve_adaptive_k(req, max_k=4) -> uses context-length heuristics
    2. Serving (spec.py): resolve_adaptive_k(req, base_k) -> base_k

    Args:
        req: Request object with device_len, remain_len attributes
        base_k: Configured --spec-mtp value (new API)
        max_k: Legacy max_k parameter for context-length heuristics
        moe_slots_per_layer: MoE cache slots per layer (unused)
        top_k_experts: Top-k experts per token (unused)

    Returns:
        Effective k for this step (0 = no speculation)
    """
    # Legacy API: context-length heuristic (used by tests)
    if max_k is not None and base_k is None:
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

    # The verify consumes k draft inputs plus the pending token. Do not allocate beyond
    # the request's remaining output budget even when serving uses a fixed configured k.
    return max(0, min(base_k or 0, req.remain_len - 1))


class _RatioWindow:
    def __init__(self):
        self.samples: deque[tuple[float, int]] = deque(maxlen=32)

    def add(self, elapsed_s: float, committed_tokens: int) -> None:
        self.samples.append((elapsed_s, committed_tokens))

    @property
    def cost(self) -> float:
        committed = sum(tokens for _, tokens in self.samples)
        return sum(elapsed for elapsed, _ in self.samples) / committed if committed else inf

    @property
    def se(self) -> float:
        n = len(self.samples)
        if n < 2:
            return inf
        committed = sum(tokens for _, tokens in self.samples)
        cost = self.cost
        residual_ss = sum((elapsed - cost * tokens) ** 2 for elapsed, tokens in self.samples)
        return sqrt(n * residual_ss / ((n - 1) * committed**2))

    def drifted(self) -> bool:
        n = len(self.samples)
        half = n // 2
        if half < 4:
            return False
        samples = list(self.samples)
        older, newer = _RatioWindow(), _RatioWindow()
        older.samples.extend(samples[:half])
        newer.samples.extend(samples[-half:])
        return abs(older.cost - newer.cost) > 2 * sqrt(older.se**2 + newer.se**2)


class AdaptiveMtpController:
    """Choose a safe MTP depth from measured seconds per committed token.

    The caller must keep each calibration probe contiguous: first k=0, then
    increasing positive depths. Once a request enters k=0 fallback, it stays
    there until ``begin_request`` starts a different request.
    """

    def __init__(self, safe_max_k: int):
        if type(safe_max_k) is not int or not 0 <= safe_max_k <= 4:
            raise ValueError("safe_max_k must be an integer from 0 through 4")
        self.safe_max_k = safe_max_k
        self._epoch = None
        self._request_uid = None
        self._stats = [_RatioWindow() for _ in range(safe_max_k + 1)]
        self._plan: deque[int] = deque()
        self._selected_depth = 0
        self._terminal_k0 = False
        self._needs_reprobe = True
        self._observations_since_check = 0
        self._discard_partial_stats = False

    @property
    def selected_depth(self) -> int:
        return 0 if self._terminal_k0 else self._selected_depth

    @property
    def probing(self) -> bool:
        return bool(self._plan) and not self._terminal_k0

    @property
    def cost_summaries(self) -> dict[int, dict[str, float | int]]:
        return {
            k: {
                "samples": len(window.samples),
                "seconds_per_token": window.cost,
                "standard_error": window.se,
            }
            for k, window in enumerate(self._stats)
        }

    def _new_epoch(self, epoch) -> None:
        self._epoch = epoch
        self._stats = [_RatioWindow() for _ in range(self.safe_max_k + 1)]
        self._needs_reprobe = True
        self._discard_partial_stats = False

    def _best_depth(self) -> int:
        baseline = self._stats[0]
        if len(baseline.samples) < 8:
            return 0
        eligible = [0]
        for depth in range(1, self.safe_max_k + 1):
            window = self._stats[depth]
            if (
                len(window.samples) >= 4
                and window.cost + 2 * window.se < baseline.cost - 2 * baseline.se
            ):
                eligible.append(depth)
        return min(eligible, key=lambda depth: self._stats[depth].cost)

    def begin_request(self, request_uid, epoch) -> None:
        if request_uid == self._request_uid and epoch == self._epoch:
            return
        incomplete_probe = bool(self._plan)
        if epoch != self._epoch:
            self._new_epoch(epoch)
        elif incomplete_probe or self._discard_partial_stats:
            self._stats = [_RatioWindow() for _ in range(self.safe_max_k + 1)]
            self._needs_reprobe = True
            self._discard_partial_stats = False
        self._request_uid = request_uid
        self._terminal_k0 = False
        self._observations_since_check = 0
        if self._needs_reprobe:
            self._plan = deque(
                [0] * 8 + [depth for depth in range(1, self.safe_max_k + 1) for _ in range(4)]
            )
            self._needs_reprobe = False
            self._selected_depth = 0
        else:
            self._plan.clear()
            self._selected_depth = self._best_depth()
            if self._selected_depth == 0:
                self._terminal_k0 = True

    def next_depth(self) -> int:
        if self._terminal_k0:
            return 0
        return self._plan[0] if self._plan else self._selected_depth

    def observe(self, depth: int, elapsed_s: float, committed_tokens: int) -> None:
        if type(depth) is not int or not 0 <= depth <= self.safe_max_k:
            raise ValueError("depth must be an integer within safe_max_k")
        if not isinstance(elapsed_s, (int, float)) or not isfinite(elapsed_s) or elapsed_s <= 0:
            raise ValueError("elapsed_s must be finite and positive")
        if type(committed_tokens) is not int or committed_tokens <= 0:
            raise ValueError("committed_tokens must be a positive integer")
        was_probing = bool(self._plan)
        if self._plan:
            if self._terminal_k0 or depth != self._plan[0]:
                raise ValueError(f"expected calibration depth {self.next_depth()}, got {depth}")
            self._plan.popleft()
        elif depth != self.next_depth():
            raise ValueError(f"expected selected depth {self.next_depth()}, got {depth}")
        window = self._stats[depth]
        window.add(float(elapsed_s), committed_tokens)
        calibration_finished = was_probing and not self._plan
        if calibration_finished:
            self._observations_since_check = 0
            if window.drifted():
                self._needs_reprobe = True
                if depth > 0:
                    self.fallback_to_k0()
            if not self._terminal_k0:
                self._selected_depth = self._best_depth()
                if self._selected_depth == 0:
                    self.fallback_to_k0()
        elif not was_probing:
            self._observations_since_check += 1
            if self._observations_since_check >= 8:
                self._observations_since_check = 0
                if window.drifted():
                    self._needs_reprobe = True
                    if depth > 0:
                        self.fallback_to_k0()
                if not self._terminal_k0:
                    self._reselect_with_hysteresis()

    def _reselect_with_hysteresis(self) -> None:
        best = self._best_depth()
        if best == 0:
            self.fallback_to_k0()
            return
        current = self._stats[self._selected_depth]
        candidate = self._stats[best]
        current_still_safe = (
            len(current.samples) >= 4
            and current.cost + 2 * current.se < self._stats[0].cost - 2 * self._stats[0].se
        )
        if not current_still_safe:
            self.fallback_to_k0()
        elif best == self._selected_depth or (
            candidate.cost + 2 * candidate.se < current.cost - 2 * current.se
        ):
            self._selected_depth = best

    def fallback_to_k0(self) -> None:
        """Lock k=0 through request end; re-probe only at the next request."""
        if self._plan:
            self._needs_reprobe = True
            self._discard_partial_stats = True
        self._terminal_k0 = True
        self._selected_depth = 0
        self._plan.clear()
