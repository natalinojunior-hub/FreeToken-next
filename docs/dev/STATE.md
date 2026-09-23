# State — 2026-09-23

Doing: nothing (stopped by operator). Evidence: /models/desenvolvimento/ft-campaign2/{RELATORIO.md,LEDGER.md} (Campaigns 3-4).
Done:
- Block 1 (e42ac55): turbo3 KL 0.97 @4098 = cuBLAS bf16 M=1 vs M>=2 only; no state defect.
- Block 2: hit/miss copy overlap NO-GO (gathers 19.3 ms vs routed GEMV 4.0 ms per k1 cycle).
- Block 3 (841e8fa, 815f1eb): QSA replan bug fixed; MTP draft graph ON by default for qwen4_exp MTP k1 + verify graph (FREETOKEN_DRAFT_GRAPH=0 disables); +2.9% k1 TG (4K 41.65, 15.7K 44.60), identical output.
- Tuning fixes: af2651d (default first on ties), f90a5e2 (paired prompts), 14e0478 (profiles written with current schema). 16K turbo3 profile = k1 + draft graph; restart without --spec-mtp uses it.
- make ci 2103 passed at 14e0478.
Decisions: tune profile schema 2 (old draft_graph=0 profiles invalid). Expert tracer (ft-campaign2/e1) archived unreviewed, not in tree.
Next:
1. Review/commit e1 tracer; one 4K cold trace (graphs off, not timed); e1/cachesim.py LRU vs byte-Belady vs candidates before coding any policy.
2. >100 tok/s not met: cycle is PCIe-bound (~1 GB missed expert bytes per k1 cycle).
