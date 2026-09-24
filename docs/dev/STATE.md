# State — 2026-09-23

## MTP depth diagnostic handoff

Production remains on k1; this diagnostic did not remeasure or change the production profile. Acceptance survivors: k1 31/63; k2 q1=27/54, q2=14/27; k3 q1=26/53, q2=13/26, q3=3/13. Expert misses per SPEC committed token (not all request tokens): k1/k2/k3 = 0.549/0.646/0.791 GB. Counters synchronize each cycle. Optimistic k2 cost is 55.27 ms versus a 48.8 ms break-even; k2+ is NO-GO. k4, k5, and 16K probes were not run for economic reasons. CPU coverage uses fake target/draft outputs, replay, and a scalar pool-slot control-flow fake; it does not prove Qwen/GDN/PLE numerical parity. EOS/cancel integration remains untested. Evidence: `/models/desenvolvimento/ft-campaign2/mtp-depth/k2-correctness.md` and `k2-correctness.log`. Current integrated validation record: `/models/desenvolvimento/ft-campaign2/mtp-depth/ci-final.log`; consult its recorded exit status. The CI result cited below is historical. Retain k1 and evaluate routing-compatible ways to lower target-pool expert bytes using the existing 4K cost model before implementation or GPU A/B.

Doing: activation complete; expert-cache policy NO-GO. Evidence: Campaign 5
ledger and expert pool analysis under `/models/desenvolvimento/ft-campaign2/e1/`.

- The final 16K GGUF/turbo3 schema-2 profile `3528c0705928df183b3d878c` selects
  MTP k1 and the draft graph: 41.04 committed tok/s, cold PP 2077 tok/s.
  A normal restart without `--spec-mtp` loaded k1 and captured verify/draft
  graphs. `FREETOKEN_DRAFT_GRAPH=0` disables draft capture. Three schema-1
  profiles with graph disabled remain invalid.
- Commits `a96d22a` and `f75f717` add opt-in eager expert tracing and repair
  the server readiness gate. Tracing is diagnostic, not a TG benchmark.
- Actual 2,886-slot LRU misses: 137.57 GB over 149 cycles at 4K and 129.46 GB
  over 145 cycles near 16K. Replay matches within 0.04%; hindsight Belady
  saves 38-42% bytes but requires future routing. Fixed-budget pool shifts
  increased 4K misses. No cache-policy change or production A/B followed.
- Residual bottleneck: 0.89-0.92 GB expert misses and 17.8-18.4 ms copy per
  k1 cycle. The optimistic Belady cost model reaches only about 50/55 tok/s
  at 4K/16K; the >100 committed tok/s goal remains unmet.
- Final integrated CI command: `TMPDIR=/models/desenvolvimento/tmp make ci`.
  Its result is stored in `/models/desenvolvimento/ft-campaign2/e1/ci-final.log`.

Next: evaluate a routing-compatible way to lower target-pool expert bytes per
committed token. Use a 4K cost model before any implementation or GPU A/B.
Do not retry unchanged LFU/ghost, CPU hybrid, or copy/compute overlap.
