# ROADMAP — freetoken-next

Implementation-first. Each step is SOURCE AUDIT → MINIMAL DESIGN → IMPLEMENT →
CORRECTNESS → 16K/32K A/B → KEEP/REVERT → NEXT.

| # | Phase | Status | Gate to leave the phase |
|---|---|---|---|
| 1 | Lineage + durable docs | **done** | `git fetch` proves base is current tip; docs exist |
| 2 | Build + import + `ft --version` on host | **done** | editable `[accel]` install, 3 C++ exts compiled, `pytest -m "not slow"` green |
| 3 | Phase 1 baselines (EXP-001/001b) | **done** | guards: 35B-A3B ≥4600 PP/≥158 TG; Flash ≥1850 PP/≥28.5 TG |
| 4 | Source audits A1–A9 | **done** | A1–A6 merged into ARCHITECTURE.md §2–§6; A7 (GGUF MoE geometry + refused-today matrix + stride-vs-file table), A8 (turbo codec bytes), A9 (unaccounted VRAM consumers) feed the rows below |
| 5 | Phase 6 VRAM ledger → governor | **done (bricks 1–3)** | `engine/vram_ledger.py` owns ceiling, reserve, overhead, the expert/KV split and the context rows; every engine prints the account and the plan; guards green at the default `--memory-ratio` (D-014, D-015, EXP-006/008) |
| 6 | 128K / 256K working, long-context arithmetic | **done** | 35B-A3B buys 128K (TG 89.3) and 256K (TG 63.8) out of the expert cache with `--kv-reserve-context` (PERFORMANCE.md §8); 512K/1M are refused by the checkpoint's 262 144-position RoPE table *and* priced unaffordable at BF16 KV (D-016) |
| 7 | Phase 2 GGUF loader (I1: `qwen3_5_moe` adapter + type tables) | **in flight** | Ornith-1.5-35B-A3B GGUF serves, shapes/dtypes asserted, sane tokens, PP/TG unregressed — A7 §7 says it currently refuses on mixed `down` geometry, so this row is gated by #9 |
| 8 | Phase 3 Turbo3/Turbo4 KV backend | pending | round-trip oracle green + VRAM/token + TG delta vs BF16 KV at 16K/32K; A8 fixes the bytes (turbo3 138 B / turbo4 132 B per 256-element row, scale is a runtime InnerQ constant set, K is not rotated, TCQ exists only on turbo4 at 127 B) |
| 9 | Phase 7 exact-geometry expert cache | pending | per-layer pools keyed by (bank, role, type); unblocks the four MoE GGUFs A7 lists as refused, and needs the row readers they lack (Q4_K/IQ4_XS/Q3_K/Q2_S are custom unsloth-v1 layouts, not llama.cpp's) |
| 10 | TCQ / VBR | pending | same gates as Turbo + per-(layer,side) tier schedule honoured |
| 11 | Phase 9 qwen4exp MTP | pending | correct acceptance/rollback; effective TG reported |
| 12 | Phase 10 MTP + TurboKV fused verify | pending | no materialize-path penalty (LTO: −13…−15 % TG) |
| 13 | Phase 8 D2D expert reuse | pending | H2D bytes avoided vs sync cost, on/off A/B |
| 14 | Phase 5 tiered/paged RAM KV | **gated out for decode** | D-016: a full-context RAM tier costs 47-462 ms of PCIe per generated token (2.2-2.7 tok/s ceiling); keep RAM for pinned expert banks, PLE and *prefix* streaming into KV, and revisit only for a bounded-read (sparse/windowed) tier |
| 15 | Adaptive MTP + full context governor UX (`--context auto`) | pending | `--kv-reserve-context` is the explicit half; the automatic half still needs a TG model for the surviving expert cache (D-015) |
| 16 | 512K/1M certification (memory physics first) | blocked upstream | needs a rope-scaled checkpoint (§8 shows the KV itself is affordable only compressed) |
| 17 | **Certification matrix green** (`benchmarks/cert_matrix.py`) | declared (D-012) | every native `-FT`/NVFP4 row clears its guard **and** every same-arch GGUF row reports parity vs that row; BLOCKED rows must name a blocker that is then fixed, never deleted |
| 18 | Close the matrix gaps | pending | shard joining + `qwen4exp` GGUF adapter + PLE-table mapping (unblocks both Flash GGUFs), a native dense `qwen35` counterpart for the 27B GGUF row, and a native-vs-KV-format A/B table |

Regression anchors that gate every row (measured on this tree, PERFORMANCE.md §3/§7):
35B-A3B **PP ≥ 4600 / TG ≥ 158**; Flash-Next **PP ≥ 1850 / TG ≥ 28.5**, both now at the default
`--memory-ratio 0.9`, with output sha1 compared as well as throughput.
