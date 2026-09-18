# STATE
Phase 1 (Bug A GDN Equivalence) DONE: EXP-036 confirmed. `gdn_decode_fla` vs `gdn_prefill_chunk_fla`
differs by ~1.95e-3 in output, ~1.43e-3 in recurrent state, ~1.95e-3 in conv state for chunks T>1,
causing greedy argmax flips on ~1.5% of steps. Unit test `test_decode_prefill_gdn_kernel_inequivalence`
added to `tests/models/qwen4_exp/test_gdn.py`.
Phase 2 (TG Attribution) NEXT: Run live cold vs warm probe (Action A) to test Hypothesis 2.1
(MoE prefill materialized expert bank copy vs decode cache).
Phase 3 (Bug B carrier probe B1-B4) READY to execute after Phase 2.
Phase 4 (Fixes: defasagem helper, Bug B carrier fix, MoE decode-path dispatch) planned.
Phase 5 (k=2/3 validation), Phase 6 (PP benchmark + 128k/256k gate), Phase 7 (fused verify spec) pending.
Baseline suite: 1934 passed / 206 skipped / 1 failed (flashinfer env).
