# State — 2026-09-23

Doing: Campaign 2 docs written (docs/dev/PERFORMANCE.md, LESSONS.md, DECISIONS.md); `ft tune` 16K result still PENDING (orchestrator to fill).
Done: GGUF beats native NVFP4-Radix end-to-end (TG 2.06x at k1) — native kept as fallback only. Offload beats hybrid on GGUF (hybrid 3.5x slower, IQ3_S CPU kernel bound) — offload stays default. Cold-prefill non-determinism fixed (bf16 index_add_ atomic order, 211efb6); verify-window vs decode KL traced to cuBLAS M=1-vs-M>=2 row divergence, not a correctness bug.
Decisions: see docs/dev/DECISIONS.md D-024..D-028 (GGUF primary, offload default, bench-bw threshold kept, deferred replay kept, profile key shape).
Next:
1. compute-sanitizer memcheck on draft-graph v2 replay to find the illegal-access kernel (QSA draft-slot addressing suspected).
2. Add hit/miss overlap to the verify-window miss path (currently serialized, no cross-layer prefetch).
3. IQ3_S AVX-512 CPU kernel throughput, only if hybrid strategy is revisited.
4. 128K/256K runs stay gated on hitting >100 tok/s first (not met yet).
