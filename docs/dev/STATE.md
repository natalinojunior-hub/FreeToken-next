# State — 2026-09-24 (campaign 9 closed; prior full state at `git show 2ab0db9:docs/dev/STATE.md`)

Done: campaign 9 k1 economics closed NO-GO. Evidence `/models/desenvolvimento/ft-campaign2/campaign9/LEDGER.md`, PERFORMANCE.md last entry.
Done: cold 4K/256 k1 traced run: hash `91e6de9c85b2`, 90/165 accepted; draft top-1 prob strongly predicts acceptance (<0.3: 23%, ≥0.9: 100%).
Decision: confidence gate at p=0: oracle +4.8%, best realistic −1.3% vs k0 → NO-GO.
Decision: per-segment k0/k1 switch oracle W=4 +5.1% with future knowledge, zero switch cost → NO-GO.
Decision: +5% would need verify −4.5 ms (−14%); all remaining MoE/replay levers are recorded no-gos. k0 stays default.
Decision: the `_adaptive_mtp_controller` gate in `spec.py` is dead (never instantiated) and unsafe (early return skips QSA/PLE/linear/residual restore). Left untouched because no gate ships.
Uncommitted: opt-in token trace (`python/freetoken/debug/`, scheduler/spec hooks, bench `--token-trace`) + trace-only `draft_prob`; `make ci` not rerun.
Next: k1 pays only with ≈68% acceptance → needs a better MTP head or a cheaper 2-row verify mechanism not yet on the no-go list.
