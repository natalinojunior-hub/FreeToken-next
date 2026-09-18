# STATE
Phase 1 (Bug A GDN Equivalence) DONE: EXP-036 confirmed. GDN decode vs prefill differs ~1.95e-3.
Phase 2 (TG Attribution) DONE: EXP-037 confirmed MoE materialization is 96.3% of forward time.
Phase 3 (Bug B Carrier Probe) DONE: EXP-038 isolated carrier as QSA pending_ring and cmp_k scratch.
Conv and recurrent states are bit-identical across requests.
Phase 4 (Fixes) NEXT:
1. Helper for lag convention (Rule 5) with unit test.
2. Fix Bug B: clear pending_ring / cmp_scratch rows on request release and spec rewind.
3. Fix TG: dispatch micro-batch spec forwards (T<=k+1) through MoE decode resident cache.
Phase 5 (k=2/3 validation), Phase 6 (PP + long-context gate), Phase 7 (verify spec) pending.
Baseline suite: 1934 passed / 206 skipped / 1 failed (flashinfer env).
