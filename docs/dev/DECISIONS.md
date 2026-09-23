# DECISIONS — freetoken-next

**Append-only.** One decision per entry; supersede rather than rewrite. Source: `old/docs/freetoken-next/DECISIONS.md` (360 lines).

---

## D-001 — Base = current upstream tip (`cac247a` = v0.1.3)
**Status:** accepted | **Why:** 23 skipped commits contain QuantConfig/Scheme/Method layers, checkpoint→reader handoff, qwen4_exp block-fp8, paged-KV granularity, Flash-Next PLE table — all needed for later phases.

## D-002 — Dedicated editable env (.venv); reference stays read-only
**Status:** accepted | **Why:** A/B against 0.1.2 anchor must remain possible.

## D-003 — Local research fork: commit locally, never push
**Status:** accepted | **Why:** Upstream AGENTS.md bars autonomous agents from push/PR.

## D-004 — GGUF immutable source of truth; no whole-file dequant at load
**Status:** accepted | **Why:** User corpus shared with other engines; derived cache/overlay only.

## D-005 — KV formats via existing quant abstraction (upstream PR #408 seams)
**Status:** revised | **Why:** A1 found no KV quant seam on cac247a; port #408's `--kv-cache-dtype` → `EngineConfig.kv_quant` → capability gate `supports_{kv_quant}_kv`; cost model: `kv_storage_bytes_per_elem` + `kv_scale_bytes_per_token`.

## D-006 — Preserve 0.1.3 prefill behaviour; replace only with causal evidence
**Status:** accepted

## D-007 — Flash-Next regression config: `--memory-ratio 0.86` (until Phase 6)
**Status:** accepted | **Why:** Default 0.9 OOMs in unbudgeted transients (Triton autotune, graph capture).

## D-008 — GGUF phase order: widen → adapt → shard
**Status:** accepted | **Why:** Every increment must be servable/measurable; loader + mmap reader + vendored kernels already exist.

## D-009 — Phase 2 sources: port upstream PR #131 (feat/generic-gguf), don't merge
**Status:** accepted | **Why:** 7 conflicts due to QuantConfig refactor; port leaf files, re-author 7 seams against HEAD.

## D-010 — KV quantization: two templates + measured warning
**Status:** accepted | **Why:** FreeToken-Kai already ships `--kv-cache-dtype {auto,q8_0,q4_0}` on our base.

## D-011 — Benchmark harness uses `--cache-type naive` (prefix cache fakes PP)
**Status:** accepted

## D-012 — Certification matrix gate: `benchmarks/cert_matrix.py`
**Status:** declared | Every native `-FT`/NVFP4 row clears guard + same-arch GGUF parity.

## D-013 — VRAM ledger single source of truth (ceiling, reserve, expert/KV split, context rows)
**Status:** implemented | `engine/vram_ledger.py` + `cache_budget.ceiling_bytes`

## D-014 — `--kv-reserve-tokens` explicit floor for context; auto-sizer rejected
**Status:** implemented | Default auto gave 8268 KV (<16K) leaving 1.08 GiB unspent.

## D-015 — VRAM governor owns expert/KV split; measured peak ratchets reserve floor
**Status:** implemented | `VramLedger.decide()` + `ContextDemand.plan_for_context()`

## D-016 — 512K/1M blocked by checkpoint RoPE (262,144 pos) + BF16 KV
**Status:** accepted | 512K = 10 GiB KV (entire pool), 1M = 20 GiB.

## D-017 — Compressed KV = capacity lever (256K+), NOT 16K speed lever
**Status:** accepted | Triton uncompressed already 9.1% TG at 16K; 256K: 5427 vs 3183 expert slots.

## D-018 — Packed layout wins only if reader stays out of address business
**Status:** accepted | Byte-per-element + gather centroid doubled ITL; fix: load row once, split in registers.

## D-019 — Turbo3/Turbo4 backend = differentiator (upstream issue #141 requests TurboQuant)
**Status:** accepted

## D-020 — FreeToken-Kai already merged our base; port-with-provenance > reinvent
**Status:** accepted | 191 commits, +31.8k lines, GGUF/KV-quant/host-bank/long-context/VRAM-accounting.

## D-021 — `qwen4_exp` MTP tensors exist in checkpoints; upstream loader drops `mtp.*` (#421)
**Status:** accepted | Phase 9 greenfield; refs/pr/69 reference loop shape.

## D-022 — MTP + TurboKV fused verify formally specified (Pillar 1 + Pillar 2)
**Status:** specified | Pillar 1: In-SRAM FP4 dequant (EXP-042). Pillar 2: Zero-replay GDN.

## D-023 — Split-kernel Turbo4/QSA: decompress kernel separate from attention
**Status:** implemented | Avoids ptxas host-RAM explosion (70 GiB); `block_n=32`, `num_stages=1`.

## D-024 — GGUF is the primary format for Qwen3.8 Flash-Next; native NVFP4-Radix kept as fallback only
**Status:** accepted | GGUF TG 2.06x native at k1, 1.34x at k0 (native's +10% PP at 4K does not
offset it). No further native specialization or 16K/3-repeat native runs planned. Campaign 2,
`ft-campaign2/LEDGER.md`.

## D-025 — `offload` is the default MoE strategy for this checkpoint; hybrid only by measured profile
**Status:** accepted | GGUF hybrid measured 8% slower than offload with the benched fetch split (37.05 vs 40.48; 3.5x before d52f287's lookup fix) (IQ3_S CPU kernel 8-11 GB/s
bound + per-layer CPU/GPU handshake overhead), after fixing the per-layer format bug (7bbf2af).
Hybrid stays available behind `--moe-strategy hybrid` / a future profile recommendation, never
auto-selected without a measured win on this hardware.

## D-026 — `ft bench bw` hybrid-vs-offload threshold (2.0x CPU/PCIe bandwidth ratio) retained as-is
**Status:** accepted | Calibrated against end-to-end measurements, not bandwidth alone (bandwidth
predicts a hybrid win at 61 > 53.6 GB/s aggregate; measured e2e is -71%, so per-layer launch
latency dominates raw bandwidth). Consistent with every measured e2e point: GGUF hybrid -71%,
native hybrid historic +3%.

## D-027 — Deferred spec-decode replay kept; `FREETOKEN_SPEC_DEFER_REPLAY=0` stays a fallback, not default-off
**Status:** accepted | Bisected to a cuBLAS bf16 non-row-independence property (M=1 vs M>=2 differ
by up to 2.4e-3), not a defer-specific error class — deferred replay's KL stays within the same
envelope as ordinary verify windows. `FREETOKEN_SPEC_DEFER_REPLAY=0` remains available for
debugging/isolation, not required for correctness.

## D-028 — Tuning profiles are keyed by GPU UUID + model + KV format + context bucket + version/kernel source; explicit flags always win
**Status:** accepted | `ft tune` (af3dcdf) persists per-machine winners under this key; a stale key
is ignored rather than trusted. The launcher only fills flags the user left unset; `--spec-mtp`
left unset resolves from the stored profile, an explicit `--spec-mtp` value is never overridden.

---

**Referência completa:** `old/docs/freetoken-next/DECISIONS.md` (D-001 a D-023 detalhados com rationale, alternativas, file:line).