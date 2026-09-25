# FreeToken-next — finish KV-RAM campaign (session 16)

Role: Opus 5.5 = orchestrator/decider only. Delegate every bounded task (grep, edits with a clear spec, tests, benchmark runs, log parsing, docs) to Sonnet 5 agents (`Agent`, model=sonnet) with self-contained prompts that cap output (≤20 lines). Opus touches code only for kernel/design work Sonnet fails at. No re-reading of files already summarized here. Reply to operator in PT-BR, 1-3 lines.

## Ground truth (do not re-derive)
- Branch `next` HEAD ≥ `075ed10`: live QSA KV RAM tier (bf16/fp8/turbo4/turbo3), hot floor + heat rebalancer, `--kv-tiering auto`, safe RAM budget + refusal "o contexto pedido de X … o máximo possível é Y", planner re-solve on validation OOM, `ft bench context`, `ft history`, GGUF split-probe fix, GGUF vision via external mmproj (verified). Design + evidence: `docs/dev/KV_RAM_DESIGN.md`, `docs/dev/STATE.md`, `docs/dev/LESSONS.md`.
- Results so far: `/models/desenvolvimento/ft-campaign2/campaign15/{results,quality}.jsonl` (queue was stopped mid-run). Harness: `cert.py` (matrix + MTP ladder), `cert2.py` (formats × quality), `needle_check.py`, `vision_check.py`, `../campaign14/run.sh`.
- Key numbers (TG tok/s): ISTA k0 64K 57.1 / 128K 55.6 / 256K 43.1; ISTA k1 128K 59.1 (k2 41.0); AD k0 64K 49.0 / 128K 46.4; RAM tier fp8 beats bf16 on TG (UD 64K 47.5 vs 41.5) with equal quality (usage 20/20, needle pass 64K/128K all formats).
- GPU rule: every GPU job under `flock /models/desenvolvimento/ft-campaign2/gpu.lock`; no SIGSTOP, no `pgrep -f`/`pkill -f` (use pid files / setsid PGID); background watchdog on "no new result in 50 min". One GPU job at a time. Never `git checkout` in the main tree while jobs run from it; use worktrees.

## Scope decisions (operator)
- Model from now on: **ISTA only** (`/models/Qwen3.8-Flash-Next-ISTA-IQ3_XXS/IQ3_XXS`). AD stays as history (keep files, no new tests). Drop NVFP4, Unsloth, AD, 35B, 27B from all tests.
- MTP head: keep only `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` (best: ISTA k1 16K TG 64.99, accept 88.1%). Today it lives in `/models/Qwen3.8-Flash-Next-Unsloth-IQ4_XS/MTP/` and `campaign13/heads/*/model/MTP/` are symlinks to it.

## Tasks (in order)
1. Disk cleanup (confirm nothing running first): move `shared-Q4_K_M.gguf` into an `MTP/` dir the engine finds for ISTA (`_find_mtp_gguf_path`: `<model dir>/MTP` or `<parent>/MTP`); verify ISTA loads it (4K k1 smoke). Then delete `/models/Qwen3.8-Flash-Next-NVFP4-Radix` (126G), `/models/Qwen3.8-Flash-Next-Unsloth-IQ4_XS` (96G, after the move), other MTP head files, and `campaign13/heads/*` symlink trees; repoint campaign scripts. Operator already authorized these deletions.
2. Turbo8 (TurboQuant 8-bit, paper-faithful): 256-level Lloyd-Max codebook for the rotated-coordinate distribution (existing `CENTROIDS_4` equals Gaussian N(0,1/128) Lloyd-Max to 1e-5 — generate 256 levels the same way, `kernel/triton/turbo_kv.py`), 1 byte/code + fp16 group norm (8.125 bpv), indices via bucketize, decode via `kernel/triton/qsa/tiered.turbo_pages_to_bf16` (add BOOK8 path), RAM-tier + CLI `--kv-ram-dtype turbo8`, tests vs `turbo_kv.decode` oracle.
3. RAM-tier format policy (operator rule): ladder BF16 → FP8 → Turbo8 → Turbo4 → Turbo3, narrowing only as far as host RAM requires; drop a wider format when a narrower one is equal quality and faster (data says FP8 dominates BF16 → auto starts at FP8; if Turbo8 ≥ FP8 on quality and TG/PP, start at Turbo8). Implement in `engine/engine.py:_kv_ram_dtype`, driven by a small table of measured dominance, not hardcoded per model.
4. Bugs found (fix + unit test each):
   a. RAM budget ignores experts pinned in host RAM: NVFP4 128K passed `_check_kv_ram_budget` then was SIGTERMed by earlyoom (RSS 73 GB). Budget must be computed after expert/host allocations or include their planned bytes.
   b. `--kv-tiering auto` does not fall back to the RAM tier when the VRAM planner is infeasible (NVFP4 64K and UD k1 256K were refused with "máximo possível" although a RAM tier would fit). Auto must retry the plan with the RAM tier before refusing.
   c. Turbo RAM pages are not rebalanced (`qsa_pool.rebalance` returns early for `host_book`): add quantizing swap or document ceiling.
   d. `mm/processor.py` `get_mm_processor` reload ignores `--mmproj` override (auto-discovers); honor it.
5. Bench gap: TG over a long generation (e.g. 128K prompt, 8K decode, TG per 1K tokens) — add `--tg-curve` to `benchmarks/bench_pp_tg.py` using the existing token stamps.

## Final measurement (after 1-5; ISTA only; `--kv-tiering auto` unless noted)
- 4K anchor (ISTA hash `2e4a55acfa90`, TG ~59) must be unchanged.
- k0 at 64K/128K/256K; MTP ladder k=1,2,… per context, stop at first k with TG < k0.
- Formats at 128K/256K with forced tier: fp8/turbo8/turbo4/turbo3 — TG, PP, needle, usage eval (20 tasks).
- TG curve at 128K. Record everything to results.jsonl (reuse `cert.py`, already resumable).

## Close
Update `docs/dev/{PERFORMANCE,STATE,KV_RAM_DESIGN,LESSONS}.md` + `ft-campaign2/campaign16/LEDGER.md` with commands, hashes, per-context tables, PASS/NO-GO for ISTA; `make ci`, ruff, mypy; local commits only, never push.
