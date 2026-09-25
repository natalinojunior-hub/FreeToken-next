# State — 2026-09-24 (campaign 13 in progress; campaign 12/11 state at `git show 0e9d00c:docs/dev/STATE.md`)
## Campaign 14/15 — live KV-RAM (2026-09-25; `next` ≥ 075ed10)
Done: QSA KV RAM tier (bf16/fp8/turbo4/turbo3, auto ladder, safe RAM budget + "máximo possível" refusal), hot floor, heat rebalancer, planner re-solve on validation OOM, ft bench context, ft history, GGUF split-probe fix, GGUF vision via mmproj (image test pass, usage 20/20, TG unchanged).
Measured (TG): ISTA k0 64K 57.1 / 128K 55.6 / 256K 43.1, k1 128K 59.1; AD 64K 49.0 / 128K 46.4; fp8 RAM tier > bf16 (UD 64K 47.5 vs 41.5), quality equal (usage 20/20, needle 64K/128K pass, all formats). Data: ft-campaign2/campaign15/{results,quality}.jsonl (queue stopped mid-run).
Scope from now: ISTA only (AD kept as history, no new tests); single MTP head shared-Q4_K_M; NVFP4/Unsloth to be deleted.
Next: `prompt-next-session-kvram-finish.md` (turbo8, format ladder, RAM-budget/auto-fallback bugs, TG curve, final ISTA certification).


Anchor: UD-IQ4_XS cold graph-on k0 4096/256: PP 1467, TG 47.28, hash `3af3056b98c0` (re-verified after `b5d4243`).
Done (`b5d4243`): GGUF tokenizer control/user tokens atomic (chat prompts were corrupted: <think>/<tool_call> split); ggml Q2_0; tiled ssm_out input gather for 256-block types; split MTP head load; `find_gguf_tensor`.
Done: ISTA GSQ-RCO IQ3_XXS loads: 4K TG 59.06 (+24.9%) PP 1651; 16K TG 58.49 (+28.4%) PP 2360; slots 5555; usage eval 20/20.
Done: KV matrix — turbo3/turbo4 cost 11-17% TG at 4K/16K; capacity lever only. KV-in-RAM deferred until engine final.
Finding: 16K prefill transient (chunk 8192, 2.02 GiB) costs ~550 expert slots; static chunk cap gives +3.4% TG but -34% PP (rejected).
Decision: k0 stays default; k1 opt-in. No tests >16K until engine final. Engine stays universal (no model hardcoding).
Decision: MTP standalone Q8 head = same bytes as shared-Q8_0 (no perf gain from fc_hidden tensor).
Evidence: `/models/desenvolvimento/ft-campaign2/campaign13/LEDGER.md`.
Done: usage eval post tokenizer fix: UD 20/20 (was 12/20), AD 20/20, ISTA 20/20.
Done: MTP k1 16K: ISTA sq8 TG 63.65 (+8.8%, accept 86.8%), UD sq8 49.27 (+8.2%, 84.8%); 4K accept ~59-61%, Q4 heads best (+1.8%); k2 4K -28%.
Finding: MTP gap = PCIe: k1 verify misses 419 MB (hit 66%) vs k0 193 MB/token (hit 75.6%); +22% bytes per output token. Best: ISTA sq4 k1 16K 64.99 (+11.1%, accept 88.1%); k2/k3 16K 47.98/44.30.
Next: `/models/desenvolvimento/ft-campaign2/campaign13/CODEX-CONTINUE.md` (MTP acceptance >75% at k1/k2/k3 per head/model, pfeifferj split shards, prefill borrows expert arena).

### Task 1 continuation (2026-09-24)

### Task 2 - pfeifferj split metadata (2026-09-24)

GGUF shard discovery now reads generic `split.no` / `split.count` metadata when filenames do not use numbered llama.cpp names. `resolve_gguf_path` selects split 0 for any shard or directory. `tests/models/test_gguf_shards.py`: 10 passed; pfeifferj metadata resolves to 2 shards and 1224 total tensors.
- Completed fresh protocol runs: NVFP4-Radix k1 4K (PP 1648.2, TG 20.07, accept 45.7%, hash `43ffb5fb8c05`, no Traceback) and AD-sq4 k1 4K (PP 1593.3, TG 52.25, accept 59.4%, hash `4f9b68a86b12`, no Traceback). Both miss the >75% acceptance gate; k0 default unchanged.

Qualification update: UD-sq4 k1 16K graph-on PP 2137.3 / TG 49.10 / accept 117/138 (84.8%) / VRAM 13.84 GiB / hash `99e6c0f8c99f` / Traceback 0. AD-sq4 k1 16K PP 2258.0 / TG 56.53 / accept 105/149 (70.5%) / VRAM 14.28 GiB / hash `cbac9b4af66d` / Traceback 0. UD passes >75% acceptance; AD does not. First UD attempt overlapped NVFP4 and OOMed; clean rerun followed after GPU release.

- Campaign13 matrix extension: sequential common-protocol run initiated across UD/AD/ISTA/NVFP4 k0-k3 at 4K/16K. Added valid UD sq4 k0 4K (PP 1463.5, TG 47.14, hash 3af3056b98c0). UD sq4 k1 4K timed out in HTTP client before summary and is excluded; no engine changes.
- UD-sq4 k1 4K rerun with extended readiness/TTFT limits still failed before a benchmark summary after ~240s loading; excluded. Further new GPU jobs paused by resource guard; use existing exact logs for matrix reporting.
- Task 3 (2026-09-24): prefill-borrows-expert-arena remains deferred. The runtime has no address-stable arena alias/eviction path: `OffloadMoeCache` keeps expert arenas allocated while prefill temporaries use separate allocations. Planner accounting was intentionally unchanged pending that implementation and in-situ GPU proof.

Task 4: `_in_proj_split` now derives from `LinearGatedDeltaGroupConfig`; qwen4_exp suite passes. `out_proj_in_perm` nsys measurement remains pending a dedicated single-GPU run.

## Production-readiness decision (2026-09-24)

- **Decision: NO-GO for final production certification.** Short-context k0 and selected 16K MTP paths are validated, but the required long-context KV tiering is not implemented and the repository rule still forbids >16K engine tests until that work is proven.
- **PASS:** GGUF split metadata discovery; qwen4_exp geometry derivation; focused loader/model tests; final `make ci` (2125 passed, 206 skipped); UD 4K k0 anchor/hash preserved.
- **CONDITIONAL:** ISTA sq4 k1 16K (TG 64.99, accept 88.1%), UD sq4/sq8 k1 16K (TG ~49.1-49.3, accept 84.8%). k1 remains opt-in.
- **NO-GO:** global k1 default, NVFP4 k1/k2/k3 qualification, AD k1 acceptance gate, 64K/128K/256K certification, and KV-in-RAM tiering without address-stable arena aliasing and graph/eager parity proof.
- GPU benchmark rule: one compute process at a time; no new long benchmark is started while `nvidia-smi --query-compute-apps` is non-empty.

## KV-RAM Gate 1 (2026-09-24)

- Design review completed in [`docs/dev/KV_RAM_DESIGN.md`](KV_RAM_DESIGN.md).
- Ownership, captured CUDA addresses, synchronization boundaries, invariants,
  failure matrix and rollback path are documented.
- Gate 1 decision: **NO-GO for implementation** until a page-owner/version
  state machine, pinned host backing, bounded CUDA-event queues, graph residency
  refusal path and eager fallback are implemented and proven by CUDA round-trip
  tests. No 64K-256K certification is authorized before Gates 1-3 pass.
