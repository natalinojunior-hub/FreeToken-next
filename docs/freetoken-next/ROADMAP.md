# ROADMAP — freetoken-next

Implementation-first. Each step is SOURCE AUDIT → MINIMAL DESIGN → IMPLEMENT →
CORRECTNESS → 16K/32K A/B → KEEP/REVERT → NEXT.

| # | Phase | Status | Gate to leave the phase |
|---|---|---|---|
| 1 | Lineage + durable docs | **done** (this commit) | `git fetch` proves base is current tip; docs exist |
| 2 | Build + import + `ft --version` on host | in progress | clean editable install of `[accel]` |
| 3 | Phase 1 baselines (EXP-001) | pending | v0.1.3 reproduces anchors within noise → immutable guards |
| 4 | Source audits A1–A4 | running | findings merged into ARCHITECTURE.md |
| 5 | Phase 2 native GGUF loader | pending | one relevant GGUF served correctly, no load-time dequant |
| 6 | Phase 3 GGUF quant dispatch/kernels | pending | per-type verdict table realised, PP/TG unregressed |
| 7 | Phase 3 Turbo3/Turbo4 KV backend | pending | round-trip test vs reference + VRAM/token + TG delta |
| 8 | TCQ / VBR | pending | same gates as Turbo + variable-rate indexing proven |
| 9 | Phase 6 unified VRAM ledger → governor | pending | no two subsystems spend the same free VRAM |
| 10 | Phase 5 tiered/paged RAM KV | pending | break-even measured; no synchronous per-token PCIe |
| 11 | Phase 9 qwen4exp MTP | pending | correct acceptance/rollback; effective TG reported |
| 12 | Phase 10 MTP + TurboKV fused verify | pending | no full-decompress scratch blow-up |
| 13 | Phase 7 exact-geometry expert cache | pending | measured on RTX 5080 |
| 14 | Phase 8 D2D expert reuse | pending | H2D bytes avoided vs sync cost, on/off A/B |
| 15 | Adaptive MTP + context governor UX (`--context 262144|auto`) | pending | no magic flags required |
| 16 | 128K/256K certification, then 512K/1M feasibility (memory physics first) | pending | 2–3 runs each, mean/variation reported |

Regression anchors that gate every row: Flash PP ≥ 1500 @16K; 35B-A3B PP ≥ 4000, TG ≥ 140.
