# State — 2026-09-23

Doing: nothing in progress. Campaign 3 closed; report /models/desenvolvimento/ft-campaign2/RELATORIO.md (Campaign 3 section) + LEDGER.md.
Done:
- Block 1 (e42ac55): turbo3 KL 0.97 @4098 = cuBLAS bf16 M=1 vs M>=2 only; no state defect.
- Block 2: hit/miss overlap NO-GO (gathers 19.3 ms vs routed GEMV 4.0 ms per k1 cycle).
- Block 3 (841e8fa, 815f1eb): QSA replan bug fixed; MTP draft graph ON by default (FREETOKEN_DRAFT_GRAPH=0 disables), +2.9% k1 TG (4K 41.65, 15.7K 44.60), identical output.
- make ci 2101 passed at e71c6a2+.
Decisions: draft graph default ON (operator approved); tune profile schema 2 invalidates all old profiles.
Next:
1. `ft tune` 16K re-run (af2651d): profile = k0 + draft graph on; MTP gain was within the 3% noise margin this run (1-11% run to run), so boot without --spec-mtp stays k0.
2. >100 tok/s not met (cold 41-45): cut missed expert bytes per cycle (~1 GB, PCIe-bound).
