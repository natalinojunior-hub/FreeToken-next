"""MTP speculation depth per step (the configured ``--spec-mtp`` k in serving)."""

from __future__ import annotations

import os
from collections import deque
from math import inf, isfinite, sqrt

# Measurement instrument only (default off => the shipped operating point stays 100% automatic).
# Set FREETOKEN_MTP_FORCE_DEPTH=<k> to pin the serving speculation depth to k for a clean
# same-depth A/B, bypassing the exploration plan and drift-triggered k=0 re-probes that otherwise
# make run-to-run depth selection timing-dependent (see campaign37 lever 4b front-loading).
_FORCE_DEPTH_ENV = "FREETOKEN_MTP_FORCE_DEPTH"


def _forced_depth_or_none(safe_max_k: int) -> int | None:
    raw = os.environ.get(_FORCE_DEPTH_ENV)
    if raw is None or raw.strip() == "":
        return None
    try:
        return max(0, min(int(raw), safe_max_k))
    except ValueError:
        return None


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


# Calibration probe budget. The k=0 baseline is measured first (contiguous, so it is not
# confounded with the warmup that the positive depths then interleave over), then the positive
# depths are probed ROUND-ROBIN (_PROBE_REPEATS passes of 1..safe_max_k) rather than grouped.
# Round-robin matters for reliability: a grouped plan ([1,1,1,1,2,2,2,2,...]) confounds depth
# with time, so any warmup/cool-down drift across the probe makes later depths look
# systematically cheaper/dearer and the depth pick becomes a timing coin-flip. Interleaving
# gives every depth the same average conditions, so the measured ordering reflects the depths.
_MIN_BASELINE_SAMPLES = 8
_MIN_DEPTH_SAMPLES = 4
_PROBE_REPEATS = 4


class AdaptiveMtpController:
    """Choose a safe MTP depth from measured seconds per committed token.

    The caller must keep each calibration probe contiguous: first k=0, then
    increasing positive depths. Once a request enters k=0 fallback, it stays
    there until ``begin_request`` starts a different request.
    """

    def __init__(self, safe_max_k: int, profiled_depth: int | None = None):
        if type(safe_max_k) is not int or not 0 <= safe_max_k <= 4:
            raise ValueError("safe_max_k must be an integer from 0 through 4")
        self.safe_max_k = safe_max_k
        self._force_depth = _forced_depth_or_none(safe_max_k)
        # A depth learned by a previous serve under the SAME hardware+model+build+config
        # fingerprint (tuning.mtp_profile). Warm-starts the controller so it skips the
        # calibration probe. Ignored when a measurement force-depth is pinned (force wins) or
        # the value is out of range. Only positive depths are honored: a learned 0 means
        # "speculation hurts here", which is rare enough and cheap enough to re-probe that it
        # is never cached, so a fluke k0 lock cannot get pinned across runs.
        self._profiled_depth: int | None = None
        if (
            self._force_depth is None
            and profiled_depth is not None
            and isinstance(profiled_depth, int)
            and 1 <= profiled_depth <= safe_max_k
        ):
            self._profiled_depth = profiled_depth
        self._pending_learned_depth: int | None = None
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

    def consume_learned_depth(self) -> int | None:
        """The depth from the most recent COMPLETED calibration, then clear it (or ``None``).

        The engine persists this to ``tuning.mtp_profile`` so the next serve warm-starts and
        skips the probe. Only a fresh cold calibration produces one: a warm start from a profile
        learns nothing new, and a forced-depth measurement run never calibrates -- so neither
        re-saves. Returns each converged depth exactly once (a drift-triggered re-calibration
        will surface the newly learned depth on the next call).
        """
        learned = self._pending_learned_depth
        self._pending_learned_depth = None
        return learned

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
        """Cheapest speculation depth, biased toward actually speculating.

        The caller asked for ``--spec-mtp k>0``, so k=0 (speculation off) is a LAST RESORT:
        return it only when every positive depth is *significantly worse* than the k=0
        baseline (speculation actively hurts -- e.g. a residency cliff makes every depth
        slower than not drafting). A depth stays eligible unless its own cost lower bound
        exceeds the baseline upper bound (2-sigma), so timing noise on a short probe can
        never lock k=0 the way the old "must be significantly BETTER" gate did -- that gate
        failed on noise and abandoned speculation for the whole request (campaign37: auto
        runs locked k0 at ~74 TG while fixed-depth k4 measured ~104). Among the eligible
        depths take the lowest measured cost, breaking ties toward the deeper draft.
        """
        baseline = self._stats[0]
        if len(baseline.samples) < _MIN_BASELINE_SAMPLES:
            return 0
        eligible = []
        for depth in range(1, self.safe_max_k + 1):
            window = self._stats[depth]
            if len(window.samples) < _MIN_DEPTH_SAMPLES:
                continue
            significantly_worse = window.cost - 2 * window.se > baseline.cost + 2 * baseline.se
            if not significantly_worse:
                eligible.append(depth)
        if not eligible:
            return 0  # every positive depth is significantly worse -> speculation hurts
        # Among the eligible depths prefer the DEEPEST whose cost is not *significantly* worse
        # than the cheapest eligible depth. The user asked for --spec-mtp k (the deepest draft),
        # so a shallower depth only wins when a deeper one is PROVABLY slower (a residency
        # cliff), not when a short 4-sample probe failed to show its edge. k3 and k4 differ by
        # ~2% in true cost while a 4-sample probe carries ~10-40% noise, so plain min-cost
        # coin-flipped k3/k4 and the engine settled at ~102 TG instead of the ~104 the cap can
        # reach ("always max TG"). Ties and noise therefore break toward the deeper draft.
        cheapest = min(eligible, key=lambda d: self._stats[d].cost)
        ceiling = self._stats[cheapest].cost + 2 * self._stats[cheapest].se
        within_noise = [d for d in eligible if self._stats[d].cost - 2 * self._stats[d].se <= ceiling]
        return max(within_noise)

    def begin_request(self, request_uid, epoch) -> None:
        if request_uid == self._request_uid and epoch == self._epoch:
            return
        if self._force_depth is not None:
            # Measurement pin: no exploration plan, no drift re-probe; hold the forced depth.
            if epoch != self._epoch:
                self._new_epoch(epoch)
            self._request_uid = request_uid
            self._terminal_k0 = False
            self._plan.clear()
            self._selected_depth = self._force_depth
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
            if self._profiled_depth is not None:
                # Warm start: a profile learned under this exact hardware+model+build+config
                # fingerprint says this depth won, so skip the 24-cycle calibration probe.
                # Epoch churn during pool warmup does NOT invalidate it (the profile encodes
                # the steady-state optimum); only measured drift (in observe) or an explicit
                # FREETOKEN_MTP_PROFILE=refresh does.
                self._plan.clear()
                self._selected_depth = self._profiled_depth
            else:
                # k=0 baseline first (contiguous), then the positive depths round-robin so the
                # depth-vs-time confound does not bias which depth looks cheapest (see the
                # _PROBE_REPEATS comment above).
                self._plan = deque(
                    [0] * _MIN_BASELINE_SAMPLES
                    + [d for _ in range(_PROBE_REPEATS) for d in range(1, self.safe_max_k + 1)]
                )
                self._selected_depth = 0
            self._needs_reprobe = False
        else:
            self._plan.clear()
            if self._profiled_depth is not None and len(self._stats[0].samples) < _MIN_BASELINE_SAMPLES:
                # Still warm (no calibration has run this epoch): hold the profiled depth
                # rather than _best_depth(), which has no k=0 baseline to compare against.
                self._selected_depth = self._profiled_depth
            else:
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
        if self._force_depth is not None:
            # Measurement pin: record cost stats for logging, but never re-probe or re-select.
            self._stats[depth].add(float(elapsed_s), committed_tokens)
            return
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
                # A drift this early (a fresh _PROBE_REPEATS-sample window) is not possible
                # (drifted() needs >= 8 samples), but keep the policy uniform: drift only
                # schedules a re-probe for the NEXT request, it never abandons this one.
                self._needs_reprobe = True
            if not self._terminal_k0:
                self._selected_depth = self._best_depth()
                if self._selected_depth == 0:
                    self.fallback_to_k0()
            if self._selected_depth > 0 and not self._terminal_k0:
                # A fresh calibration just converged: expose the learned depth so the engine can
                # persist it (tuning.mtp_profile) and the next serve skip the probe. Only a
                # positive depth is cached -- a learned 0 ("speculation hurts") is rare and cheap
                # to re-probe, and never caching it means a fluke k0 lock cannot get pinned.
                self._pending_learned_depth = self._selected_depth
        elif not was_probing:
            self._observations_since_check += 1
            if self._observations_since_check >= 8:
                self._observations_since_check = 0
                if self._profiled_depth is None:
                    # Cold/calibrated: drift schedules a re-probe for the NEXT request (never a
                    # k0 lock -- that threw away ~20 TG on noise); re-selection below drops to
                    # k0 only if the depth became genuinely worse than the baseline, and keeps a
                    # depth that merely got cheaper (beneficial drift).
                    if window.drifted():
                        self._needs_reprobe = True
                    if (
                        not self._terminal_k0
                        and len(self._stats[0].samples) >= _MIN_BASELINE_SAMPLES
                    ):
                        self._reselect_with_hysteresis()
                # else: WARM start is RIGID for the whole serve. The profile is keyed on the
                # hardware+model+build+config fingerprint -- all fixed within a serve -- and the
                # resolver pins residency (compact + bf16 keep the expert pool VMM-lazy), so
                # in-serve cost movement is warmup/noise/request-transition or an LRU residency
                # change that hits every depth equally: NOT a change in which depth is optimal.
                # Reacting to it re-probed a request spuriously and cost ~3 TG of warm mean
                # (observed 100.9 vs run0 103.8). Hold the profiled depth; recalibration happens
                # on a fingerprint change (next serve) or FREETOKEN_MTP_PROFILE=refresh. A
                # genuine runtime FAILURE still falls back to k0 via spec.py's error handling,
                # so rigidity costs no safety.

    def _reselect_with_hysteresis(self) -> None:
        best = self._best_depth()
        if best == 0:
            # Every positive depth is now significantly worse than the k=0 baseline:
            # speculation stopped paying (a real regime change), so drop to k=0.
            self.fallback_to_k0()
            return
        if best == self._selected_depth:
            return  # still the cheapest eligible depth; keep it (no thrash)
        current = self._stats[self._selected_depth]
        candidate = self._stats[best]
        # Switch only on a clear win: the candidate is significantly cheaper than the current
        # depth, OR the current depth has itself become significantly worse than the baseline.
        # The old gate dropped straight to k=0 whenever the current depth was no longer
        # *significantly better* than k=0 -- on noise that abandoned a depth that was still
        # the cheapest, so hysteresis now compares against the baseline only to detect a
        # genuinely unsafe current depth, and otherwise moves to the better positive depth.
        current_unsafe = current.cost - 2 * current.se > self._stats[0].cost + 2 * self._stats[0].se
        if current_unsafe or (candidate.cost + 2 * candidate.se < current.cost - 2 * current.se):
            self._selected_depth = best

    def fallback_to_k0(self) -> None:
        """Lock k=0 through request end; re-probe only at the next request."""
        if self._plan:
            self._needs_reprobe = True
            self._discard_partial_stats = True
        self._terminal_k0 = True
        self._selected_depth = 0
        self._plan.clear()
