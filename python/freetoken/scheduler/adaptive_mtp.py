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

    def drifted(self, *, harmful_only: bool = False) -> bool:
        n = len(self.samples)
        half = n // 2
        if half < 4:
            return False
        samples = list(self.samples)
        older, newer = _RatioWindow(), _RatioWindow()
        older.samples.extend(samples[:half])
        newer.samples.extend(samples[-half:])
        if harmful_only and newer.cost <= 1.10 * older.cost:
            return False
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
# Cap for the amortized k=0 audit cadence (see observe(): doubles on each confirming audit).
_MAX_BASELINE_INTERVAL = 4096
_DEPTH_AUDIT_INTERVAL = 64
_DEPTH_AUDIT_SAMPLES = 4


class AdaptiveMtpController:
    """Choose a safe MTP depth from measured seconds per committed token.

    Each calibration samples k=0 first, then positive depths round-robin.
    Cached depths receive bounded k=0 audits. Once a request enters k=0 fallback, it stays
    there until ``begin_request`` starts a different request.
    """

    def __init__(self, safe_max_k: int, profiled_depth: int | None = None):
        if type(safe_max_k) is not int or safe_max_k < 0:
            raise ValueError("safe_max_k must be a non-negative integer")
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
        self._auditing = False
        self._cycles_since_baseline = 0
        self._baseline_interval = 32
        self._cycles_since_depth_audit = 0
        self._depth_audit_depth: int | None = None
        self._depth_audit_samples = 0
        self._depth_audit_inflight = False
        self._depth_audited: set[int] = set()

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
        if self._epoch is not None:
            self._profiled_depth = None
        self._epoch = epoch
        self._stats = [_RatioWindow() for _ in range(self.safe_max_k + 1)]
        self._needs_reprobe = True
        self._observations_since_check = 0
        self._discard_partial_stats = False
        self._auditing = False
        self._cycles_since_baseline = 0
        self._baseline_interval = 32
        self._cycles_since_depth_audit = 0
        self._depth_audit_depth = None
        self._depth_audit_samples = 0
        self._depth_audit_inflight = False
        self._depth_audited.clear()

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
        depths take the lowest measured cost, unbiased by depth.
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
        # Among the eligible depths take the lowest measured cost with NO depth bias.
        # (Previous bias took the DEEPEST depth whose noise band touched the cheapest's;
        # on a saturated-acceptance corpus one extra draft step is +27% true step cost and
        # a 4-sample band still waved it through, so auto settled on k6 at 97 TG while k5
        # measured 106. campaign37's k0-lock fear is handled by the eligibility gate above
        # (2-sigma vs baseline), not by depth preference; the hysteresis in
        # _reselect_with_hysteresis keeps the incumbent on genuine 2% coin-flips.)
        cheapest = min(eligible, key=lambda d: self._stats[d].cost)
        return cheapest

    def _next_depth_challenger(self, *, exclude: int | None = None) -> int | None:
        current = self._stats[self._selected_depth]
        for depth in range(self.safe_max_k, 0, -1):
            if depth in (self._selected_depth, exclude) or depth in self._depth_audited:
                continue
            candidate = self._stats[depth]
            if len(candidate.samples) < _MIN_DEPTH_SAMPLES:
                continue
            if candidate.cost - 2 * candidate.se <= current.cost + 2 * current.se:
                return depth
        return None

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
        elif incomplete_probe and self._auditing:
            self._request_uid = request_uid
            self._terminal_k0 = False
            return
        elif incomplete_probe or self._discard_partial_stats:
            self._stats = [_RatioWindow() for _ in range(self.safe_max_k + 1)]
            self._depth_audited.clear()
            self._needs_reprobe = True
            self._discard_partial_stats = False
        self._request_uid = request_uid
        self._terminal_k0 = False
        if self._needs_reprobe:
            if self._profiled_depth is not None:
                # A cached depth seeds the first epoch; live economics still audit it below.
                self._plan.clear()
                self._selected_depth = self._profiled_depth
            else:
                # A cold probe is a new campaign. Discard old measurements so drift-triggered
                # recalibration compares depths using samples from the current regime.
                self._stats = [_RatioWindow() for _ in range(self.safe_max_k + 1)]
                self._depth_audited.clear()
                # k=0 baseline first (contiguous), then the positive depths round-robin so the
                # depth-vs-time confound does not bias which depth looks cheapest (see the
                # _PROBE_REPEATS comment above).
                self._plan = deque(
                    [0] * _MIN_BASELINE_SAMPLES
                    + [d for _ in range(_PROBE_REPEATS) for d in range(1, self.safe_max_k + 1)]
                )
                self._selected_depth = 0
            self._needs_reprobe = False
            if self._profiled_depth is not None:
                # The profile already came from a full calibration; defer k=0 validation and
                # probe an adjacent positive depth to detect prompt-regime changes cheaply.
                self._baseline_interval = 512
                self._depth_audit_depth = (
                    self.safe_max_k
                    if self._profiled_depth < self.safe_max_k
                    else self._profiled_depth - 1
                ) or None
        else:
            self._plan.clear()
            if (
                self._profiled_depth is not None
                and len(self._stats[0].samples) < _MIN_BASELINE_SAMPLES
            ):
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
            if (self._terminal_k0 and not self._auditing) or depth != self._plan[0]:
                raise ValueError(f"expected calibration depth {self.next_depth()}, got {depth}")
            self._plan.popleft()
        elif depth != self.next_depth():
            raise ValueError(f"expected selected depth {self.next_depth()}, got {depth}")
        window = self._stats[depth]
        window.add(float(elapsed_s), committed_tokens)
        calibration_finished = was_probing and not self._plan
        if calibration_finished:
            was_auditing = self._auditing
            was_depth_audit = self._depth_audit_inflight
            depth_before_audit = self._selected_depth
            self._auditing = False
            self._depth_audit_inflight = False
            self._profiled_depth = None
            if not was_depth_audit:
                self._cycles_since_baseline = 0
                self._observations_since_check = 0
            if not self._terminal_k0:
                if was_depth_audit:
                    candidate = self._depth_audit_depth
                    if candidate is not None:
                        self._depth_audit_samples = len(self._stats[candidate].samples)
                        self._reselect_with_hysteresis()
                        if self._selected_depth != depth_before_audit:
                            self._pending_learned_depth = self._selected_depth
                            self._depth_audit_depth = self._next_depth_challenger(exclude=candidate)
                            self._depth_audit_samples = 0
                        elif (
                            self._depth_audit_samples >= 8
                            or self._stats[candidate].cost - 2 * self._stats[candidate].se
                            > self._stats[self._selected_depth].cost
                            + 2 * self._stats[self._selected_depth].se
                        ):
                            self._depth_audited.add(candidate)
                            self._depth_audit_depth = self._next_depth_challenger(exclude=candidate)
                            if self._depth_audit_depth is None and candidate > 1:
                                next_depth = candidate - 1
                                self._depth_audit_depth = (
                                    next_depth if next_depth != self._selected_depth else None
                                )
                            self._depth_audit_samples = 0
                elif was_auditing:
                    # An audit is an insurance re-check of a cached depth, not a fresh
                    # campaign: keep the current depth unless it is clearly beaten or
                    # unsafe. The old direct _best_depth() call let probe noise flip the
                    # pick every cycle, and each flip reset the amortized cadence --
                    # the whole 103.6 -> 101.6 auto deficit measured on the cert model.
                    self._reselect_with_hysteresis()
                else:
                    # Cold samples are noisy, especially at the requested maximum depth. Keep
                    # that depth unless another positive depth clearly beats it; a short probe
                    # must not replace a good requested cap with a lower point estimate.
                    self._selected_depth = self.safe_max_k
                    self._reselect_with_hysteresis()
                if self._selected_depth == 0:
                    self.fallback_to_k0()
                    self._needs_reprobe = True
            if was_auditing and not was_depth_audit:
                # Amortized counterfactual insurance: each audit that CONFIRMS the same
                # depth is new evidence of a stable regime, so the cadence quadruples
                # (capped). A changed pick resets it -- a contested choice must be
                # re-audited promptly.
                self._baseline_interval = (
                    128
                    if self._selected_depth != depth_before_audit
                    else min(4 * max(self._baseline_interval, 32), _MAX_BASELINE_INTERVAL)
                )
            elif not was_auditing:
                # A fresh campaign just measured every depth against a brand-new k=0
                # baseline; the counterfactual is proven current, so the first audit can
                # wait a full amortized interval instead of taxing the next served burst.
                self._baseline_interval = 512
            if self._selected_depth > 0 and not self._terminal_k0:
                # A fresh calibration just converged: expose the learned depth so the engine can
                # persist it (tuning.mtp_profile) and the next serve skip the probe. Only a
                # positive depth is cached -- a learned 0 ("speculation hurts") is rare and cheap
                # to re-probe, and never caching it means a fluke k0 lock cannot get pinned.
                self._pending_learned_depth = self._selected_depth
            if not was_auditing and self._depth_audit_depth is None:
                self._depth_audit_depth = self._next_depth_challenger()
        elif not was_probing:
            self._observations_since_check += 1
            if depth > 0:
                self._cycles_since_baseline += 1
                self._cycles_since_depth_audit += 1
            if self._observations_since_check >= 8:
                self._observations_since_check = 0
                if (
                    len(self._stats[0].samples) >= _MIN_BASELINE_SAMPLES
                    and window.drifted(harmful_only=True)
                    and self._best_depth() != depth
                ):
                    # A cost rise alone does not invalidate a depth that still wins the
                    # measured comparison. Periodic fresh k0 audits remain active below.
                    self._profiled_depth = None
                    self._needs_reprobe = True
                    self._depth_audited.clear()
                    # A contested regime earns its counterfactual promptly again.
                    self._baseline_interval = 64
                if not self._terminal_k0 and len(self._stats[0].samples) >= _MIN_BASELINE_SAMPLES:
                    self._reselect_with_hysteresis()
            if not self._terminal_k0 and self._cycles_since_baseline >= self._baseline_interval:
                # Measure the counterfactual even when a bad cached depth never drifts.
                self._stats[0] = _RatioWindow()
                self._plan = deque([0] * _MIN_BASELINE_SAMPLES)
                self._auditing = True
            elif (
                not self._terminal_k0
                and self._depth_audit_depth is not None
                and len(self._stats[0].samples) >= _MIN_BASELINE_SAMPLES
                and self._cycles_since_depth_audit >= _DEPTH_AUDIT_INTERVAL
            ):
                if self._depth_audit_samples == 0:
                    self._stats[self._depth_audit_depth] = _RatioWindow()
                self._plan = deque([self._depth_audit_depth])
                self._auditing = True
                self._depth_audit_inflight = True
                self._cycles_since_depth_audit = 0

    def _reselect_with_hysteresis(self) -> None:
        best = self._best_depth()
        if best == 0:
            # Every positive depth is now significantly worse than the k=0 baseline:
            # speculation stopped paying (a real regime change), so drop to k=0.
            self.fallback_to_k0()
            self._needs_reprobe = True
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
        if self._auditing:
            # Tail raw steps are valid baseline samples; retain an unfinished audit.
            self._terminal_k0 = True
            return
        if self._plan:
            self._needs_reprobe = True
            self._discard_partial_stats = True
        self._terminal_k0 = True
        self._selected_depth = 0
        self._plan.clear()
