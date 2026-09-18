# STATE
Phase 1 (Bug A GDN Equivalence) DONE: EXP-036 confirmed. GDN decode vs prefill differs ~1.95e-3.
Phase 2 (TG Attribution) DONE: EXP-037 confirmed Hypothesis 2.1. MoE accounts for 1197.45 ms
(96.3%) of the 1243.01 ms prefill forward due to 48 layers of 512-expert materialization and GEMMs.
Decode forward is 36-50 ms. Cold vs warm probe showed verify_forward is constant ~1.20s in both;
the TG jump was purely rejection (2 forwards) vs acceptance (1 forward).
Phase 3 (Bug B Carrier Probe B1-B4) NEXT: Probe state carriers across requests (QSA pending ring,
conv states, linear states, cmp_k).
Phase 4 (Fixes: defasagem helper, Bug B carrier, MoE decode-path dispatch) planned.
Phase 5 (k=2/3 validation), Phase 6 (PP + long-context gate), Phase 7 (verify spec) pending.
Baseline suite: 1934 passed / 206 skipped / 1 failed (flashinfer env).
