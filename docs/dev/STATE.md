# State Snapshot — 2026-09-22

## Doing Now
- VRAM audit DONE; relatorio-vram-definitivo.md being finalized. Nothing committed.

## Done
1. Canonical ledger in memory_planner.py (single solve, measured budget once, 2-point transient, no margins/retries, infeasible report). Tests: tests/engine/test_memory_planner_ledger.py.
2. Root causes fixed: GGUF lazy pageable experts (HMM 2.41 GiB), probe chunk extrapolation, double counts, PLE wait-sync graph deadlock (auto->gate), GGUF MTP meta embedding, turbo_kv host tensors in capture, MTP snapshot slot eviction.
3. Certification 16K, CUDA graph, auto planner: GGUF/FTW x turbo3/turbo4 PASS; MTP k=1 FTW/GGUF PASS. Reconciliation: no unexplained >128 MiB.
4. make ci PASS (CPU-only). muse_glimmer slow test marked slow.
5. Harness benchmarks/cert_matrix.py: one boot per config, event watchdog (GPU idle 2s, death, health) + py-spy native dumps.

## Decisions
- PLE disk sync default = launch-gating (wait-sync deadlocks graphs).
- Never run make ci concurrently with GPU serve (earlyoom + basetemp wipes /models/desenvolvimento/tmp).

## Next
1. MTP k=1 lowers TG (GGUF 28->10, FTW ~28->10-19): investigate. 2. Per-geometry expert slot pools (~2.5x waste GGUF). 3. Lazy GDN snapshot slots. 4. Certify 128K/256K. 5. Commit only when operator asks.
