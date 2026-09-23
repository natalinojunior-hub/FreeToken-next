# State — 2026-09-23

Doing: Campaign 3 block 3 (MTP draft-step CUDA graph): Sonnet worker reproducing the replay illegal access in worktree ft-campaign2/wt3 (compute-sanitizer), deliverables ft-campaign2/c3/REPORT.md + draft-graph-v3.patch. GPU owned by that worker.
Done:
- Block 1 CLOSED (e42ac55, ft-campaign2/A2b-outlier-4098.md): turbo3 KL 0.97 @4098 = cuBLAS bf16 M=1 vs M>=2 only (row-wise GEMMs -> bitwise k0). No code change.
- Block 2 NO-GO (ft-campaign2/B2-verdict.md): per k1 cycle gathers 19.3 ms vs routed GEMV 4.0 ms; hit/miss overlap ceiling <3 ms realistic. Not coded.
Decisions: no row-wise GEMM promotion (parity only, cost); overlap rejected on measured cost model; nsys not installed -> torch.profiler traces (no system change).
Next:
1. Review worker's root cause/fix; if valid apply v3 in wt, run CHECK=1 end-to-end + cold bench 4K/15.7K k1 (bench.sh), enable only on net gain.
2. make ci on final HEAD; update ft-campaign2/RELATORIO.md + LEDGER.md with commits/verdicts; >100 tok/s still unmet (cold 40-44).
