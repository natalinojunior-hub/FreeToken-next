# State — 2026-09-24 (campaign 11 Gate 1 closed; campaign 10 state at `git show 3f3ba92:docs/dev/STATE.md`)

Anchor: cold graph-on k0 4096/256 at `9a640ef`: PP 1466.7, TG 47.28, 20.24 ms/step = gather 7.57 + dense q8_0 5.16 + routed GEMV 1.79 + rest; slots 3519.
Done: Gate 1 offline byte model at `3f3ba92` (no source change): LRU replay of 4K trace vs slots/row bytes + measured dense MMVQ cost per quant type.
Measured: dense set time vs Q8_0 — Q6_K 0.942, Q5_K 0.706, Q4_K 0.650 (Q6_K kernel less bandwidth-efficient).
Ceilings (optimistic TG): more resident experts without new formats +1.2..+2.7% (+5.2% only by spending the memory-ratio margin) -> reject; dense Q6_K +3.6% -> reject.
Survive ceiling: AD-4.27 GGUF experts (IQ2_S gate/up) +9.5..10.5%; dense Q5_K +11.5%, Q4_K +14.1%, Q5_K + MTP-pool/embd reclaim +14.1%.
Blocked: AD-4.27 is an unlisted checkpoint (AGENTS.md model directive), IQ2_S quality unmeasured; dense requant needs a new qwen4exp converter (none on host) and carries quality risk without imatrix.
Decision: no Gate 3 A/B without operator authorization; k0 default unchanged. >100 tok/s: no measured path (best ceiling ~55).
Evidence: `/models/desenvolvimento/ft-campaign2/campaign11/GATE1.md`, `ceil.py`, `sim.py`, `mmvq_bench.py`, `ceilings.txt`.
Next: operator picks (1) authorize dense Q5_K conversion of UD-IQ4_XS, (2) authorize AD-4.27 A/B, or (3) stop; either needs a teacher-forced KL quality gate before TG A/B.
