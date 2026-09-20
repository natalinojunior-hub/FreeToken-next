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
| 7 | Phase 2 GGUF loader (I1: `qwen3_5_moe` adapter + type tables) | **done** | Ornith-1.5-35B-A3B and Tiel-Coder GGUF serve with mixed geometry unblocked by #9; shapes/dtypes asserted |
| 8 | Phase 3 Turbo3/Turbo4 KV backend | **done** | Split Turbo4 certified on Flash-Next at 16K and 128K. Bug A offline characterized (EXP-036). Bug B root-caused (EXP-038) and fixed via QSA `free_req` zeroing and spec snapshot/restore (EXP-039), bit-identical across repeats. TG bottleneck localized to MoE prefill materialization (EXP-037) and solved via MoE decode-path dispatch for spec micro-batches (EXP-040), dropping verify forward from 1.20s to 0.073s and lifting live TG from 0.79 tok/s to 19.28-25.0 tok/s (~25x speedup). Triton + BF16 baseline compared at 16K (PP 1814.7, TG 21.17, sha1 `614aa7bcdf59` matching Turbo4 + MTP bit-for-bit). Multi-token speculation validated live (k=2 and k=3). Long-context Turbo4 128K certified live (PP 1376.1 tok/s, TG 4.86 tok/s, 14.84 GiB VRAM, EXP-041); 256K ledger physics confirmed. Active proc watchdog added to benchmarks. |
| 9 | Phase 7 exact-geometry expert cache | **done** | per-layer pools keyed by (bank, role, type) — that is the whole unblock for the **two** checkpoints A7 §7 proves are blocked by geometry alone (Ornith: `gate_up` mixes Q3_K/Q4_K; Tiel: `down` mixes Q6_K/IQ4_XS), plus the per-role split for the fused `gate_up` bank. A7 §8's stride-vs-file pass closed the layout question (ids are llama.cpp's byte for byte, 1194/1194 tensors match), and **no GPU row reader is missing**: `--moe-strategy offload` dequantizes every type in `BLOCK_SHAPE`. The format gap is CPU-side only — `_cpu_moe` has dot kernels for ids 2/12/14 and `_resolve_gguf_format` (`moe/cpu_executor.py:86-119`) accepts one format for both banks, so those checkpoints need offload until Q3_K/IQ3_S/IQ4_XS get CPU kernels |
| 10 | TCQ / VBR | **done** | same gates as Turbo + per-(layer,side) tier schedule honoured (`kvcache/tcq_policy.py`) |
| 11 | Phase 9 qwen4exp MTP | **validated (EXP-043/044/045)** | 1-step residual carry shift fixed draft alignment; draft acceptance 90.9% overall (86.4% 2/2); streaming detokenizer isolated; deterministic text verified live (`output_sha1: ed45eb6cc897`) |
| 12 | Phase 10 MTP + TurboKV fused verify | **done (D-022/023)** | Pillar 1 shipped (in-SRAM FP4 dequant in Triton QSA attend kernel); Pillar 2 shipped (zero-replay GDN step-by-step checkpoint rollback) |
| 13 | Phase 8 D2D expert reuse | **done** | H2D bytes avoided vs sync cost, auto-enabled on Blackwell SM120 (`moe_prefill_hit_d2d`) |
| 14 | Phase 5 tiered/paged RAM KV | **gated out for decode** | D-016: a full-context RAM tier costs 47-462 ms of PCIe per generated token (2.2-2.7 tok/s ceiling); keep RAM for pinned expert banks, PLE and *prefix* streaming into KV, and revisit only for a bounded-read (sparse/windowed) tier |
| 15 | Adaptive MTP + full context governor UX (`--context auto`) | **done** | dynamic speculation depth scaling across context lengths (`scheduler/adaptive_mtp.py`) |
| 16 | 512K/1M certification (memory physics first) | **ready** | auto-extending RoPE table to `seq_override` positions without OOB table reads; testing deferred |
| 17 | **Certification matrix green** (`benchmarks/cert_matrix.py`) | **ready** | Flash GGUF unblocked; dry-run validates all active rows |
| 18 | Close the matrix gaps | **done** | shard joining + `qwen4exp` GGUF adapter + PLE-table mapping (unblocks Flash UD-IQ4_XS GGUF) |

Regression anchors that gate every row (measured on this tree, PERFORMANCE.md §3/§7):
35B-A3B **PP ≥ 4600 / TG ≥ 158**; Flash-Next **PP ≥ 1850 / TG ≥ 28.5**, both now at the default
`--memory-ratio 0.9`, with output sha1 compared as well as throughput.
