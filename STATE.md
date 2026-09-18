# STATE
ROADMAP Line 8 CLOSED: Turbo4 + MTP fully certified (EXP-036 to EXP-041).
Phase 1 (Bug A): GDN decode vs prefill discrepancy characterized (EXP-036).
Phase 2-3 (TG & Carrier): MoE materialization 96% cost (EXP-037); QSA carrier isolated (EXP-038).
Phase 4 (Fixes): Bug B fixed via QSA free_req zeroing & spec snapshot/restore (EXP-039).
MoE micro-batch decode dispatch lifted spec TG from 0.79 to 19.28-25.0 tok/s (~25x, EXP-040).
Phase 5 (k=2/3): Multi-token speculation validated live on Flash-Next.
Phase 6 (Baselines & Long Context): triton+bf16 sha1 (614aa7bcdf59) bit-identical to Turbo4+MTP.
Long-context 128K certified live: PP 1376.1 tok/s, TG 4.86 tok/s, 14.84 GiB VRAM (EXP-041).
Proc watchdog added to benchmarks; rotate/inv_rotate transient memory bounded.
ROADMAP Line 12 SPECIFIED: TreeWY delta-rule pseudo-values & fused verify (D-022).
Suite gate: 1938 passed, 206 skipped, 1 failed (known flashinfer env).
