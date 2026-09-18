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
| 7 | Phase 2 GGUF loader (I1: `qwen3_5_moe` adapter + type tables) | **in flight** | Ornith-1.5-35B-A3B GGUF serves, shapes/dtypes asserted, sane tokens, PP/TG unregressed — A7 §7 says it currently refuses on mixed `gate_up` geometry (`{'Q3_K': [5..34], 'Q4_K': [0..4, 35..39]}`), so this row is gated by #9 |
| 8 | Phase 3 Turbo3/Turbo4 KV backend | **in flight — MTP+Turbo4 runs live, TG severely regressed, correctness re-open** | Split Turbo4 runs on Flash-Next at PP 1713.7 / TG 24.72 in eager/no-overlap mode. MTP=1's page leak (EXP-033) was actually three bugs in `spec.py` (EXP-034), all fixed and regression-tested; live now completes with no crash: PP 1715.2 (matches anchor) but **TG 0.79 mean vs 24.6 without MTP (31-50x slower)**, tied to `--cuda-graph-max-bs 0` forcing eager per-step Triton launches -- CUDA graph support for `--spec-mtp` is the next blocker (goal priority 4). Also found: multi-request same-session output diverges (`output_sha1` differs across repeats) though a single isolated request is deterministic -- a NEW, unfixed cross-request state-leak bug; do not trust multi-request MTP+Turbo4 content-correctness until root-caused. Gate remains matched `triton + bf16`, repeated Turbo4 16K, then 128K/256K. |
| 9 | Phase 7 exact-geometry expert cache | pending | per-layer pools keyed by (bank, role, type) — that is the whole unblock for the **two** checkpoints A7 §7 proves are blocked by geometry alone (Ornith: `gate_up` mixes Q3_K/Q4_K; Tiel: `down` mixes Q6_K/IQ4_XS), plus the per-role split for the fused `gate_up` bank. A7 §8's stride-vs-file pass closed the layout question (ids are llama.cpp's byte for byte, 1194/1194 tensors match), and **no GPU row reader is missing**: `--moe-strategy offload` dequantizes every type in `BLOCK_SHAPE`. The format gap is CPU-side only — `_cpu_moe` has dot kernels for ids 2/12/14 and `_resolve_gguf_format` (`moe/cpu_executor.py:86-119`) accepts one format for both banks, so those checkpoints need offload until Q3_K/IQ3_S/IQ4_XS get CPU kernels |
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
