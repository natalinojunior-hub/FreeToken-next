# ROADMAP — freetoken-next

Implementation-first. Each step is SOURCE AUDIT → MINIMAL DESIGN → IMPLEMENT →
CORRECTNESS → 16K/32K A/B → KEEP/REVERT → NEXT.

| # | Phase | Status | Gate to leave the phase |
|---|---|---|---|
| 1 | Lineage + durable docs | **done** | `git fetch` proves base is current tip; docs exist |
| 2 | Build + import + `ft --version` on host | **done** | editable `[accel]` install, 3 C++ exts compiled, `pytest -m "not slow"` green |
| 3 | Phase 1 baselines (EXP-001/001b) | **done** | guards: 35B-A3B ≥4600 PP/≥158 TG; Flash ≥1850 PP/≥28.5 TG |
| 4 | Source audits A1–A6 | **done** | merged into ARCHITECTURE.md §2–§6; reports in `audits/` |
| 5 | Phase 2 GGUF loader (I1: `qwen3_5_moe` adapter + type tables) | **next** | Ornith-1.5-35B-A3B GGUF serves, shapes/dtypes asserted, sane tokens, PP/TG unregressed |
| 6 | Phase 2 I2/I3: ggml expert banks, shard joining | pending | 3-shard `qwen4exp` target opens; Q4_K/Q6_K banks in host banks |
| 7 | Phase 3 Turbo3/Turbo4 KV backend (on #408's seams) | pending | round-trip oracle green + VRAM/token + TG delta vs BF16 KV at 16K/32K |
| 8 | TCQ / VBR | pending | same gates as Turbo + per-(layer,side) tier schedule honoured |
| 9 | Phase 6 unified VRAM ledger → governor | pending | no two subsystems spend the same free VRAM; transients budgeted (D-007 becomes obsolete) |
| 10 | Phase 5 tiered/paged RAM KV | pending | break-even measured; no synchronous per-token PCIe |
| 11 | Phase 9 qwen4exp MTP | pending | correct acceptance/rollback; effective TG reported |
| 12 | Phase 10 MTP + TurboKV fused verify | pending | no materialize-path penalty (LTO: −13…−15 % TG) |
| 13 | Phase 7 exact-geometry expert cache | pending | measured on RTX 5080 |
| 14 | Phase 8 D2D expert reuse | pending | H2D bytes avoided vs sync cost, on/off A/B |
| 15 | Adaptive MTP + context governor UX (`--context 262144|auto`) | pending | no magic flags required |
| 16 | 128K/256K certification, then 512K/1M feasibility (memory physics first) | pending | 2–3 runs each, mean/variation reported |

Regression anchors that gate every row (measured on this tree, PERFORMANCE.md §3):
35B-A3B **PP ≥ 4600 / TG ≥ 158**; Flash-Next **PP ≥ 1850 / TG ≥ 28.5** at `--memory-ratio 0.86`.
