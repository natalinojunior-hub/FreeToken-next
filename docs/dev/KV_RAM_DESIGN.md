# KV RAM tiering design gate

Status: **implemented for QSA BF16; on by default (`auto`) for certified families** (2026-09-25, campaigns 14-16, branch `next`). Gate 1 text below is historical.

This document records the Gate 1 review for host-RAM KV placement. It is a
design boundary, not an approval to change the allocator.

## Current ownership and lifetime

```text
Engine
  -> KV pool (MHA/QSA/Turbo/DSA family)
       -> page tensors on CUDA; page table [request, token] on CUDA
       -> attention metadata and out locations rebuilt for each Batch
  -> LinearStatePool
       -> GDN conv/recurrent state and PLE slot state on CUDA
       -> live, ping-pong, and committed snapshot slots
  -> OffloadMoeCache
       -> expert arenas and LRU metadata; arenas stay allocated during prefill
  -> GraphRunner
       -> decode/verify/draft CUDA graphs and capture pools
       -> graph input, output, page-table, and GDN index buffers at fixed addresses
Scheduler/spec
  -> snapshots QSA, PLE, GDN residual and speculative page ownership
  -> accept/rollback/replay; request cancellation returns table/page/slot ownership
```

## Captured addresses and synchronization

- `GraphCaptureBuffer` owns fixed CUDA tensors for input IDs, positions,
  output locations, table indices, logits and FLA metadata.
- `VerifyGraph` and `DraftGraph` copy request metadata into those buffers and
  replay captured kernels. The graph consumes CUDA page-table addresses,
  `out_loc`, `cache_indices`, QSA metadata and GDN/PLE slot state.
- Graph capture uses a shared CUDA graph pool. Rebinding or reallocating any
  tensor reachable by a captured kernel invalidates the graph contract.
- Scheduler state changes happen around forward/replay boundaries; rollback
  restores QSA/PLE/GDN snapshots and speculative page ownership before the next
  request step.
- Existing cache rebuilds are identity-preserving only for supported in-VRAM
  pools and are rejected before destructive free when the target does not fit.

## Tiering contract

1. Hot KV pages and every graph-bound buffer keep stable CUDA virtual addresses.
2. A cold page has one owner and a version. Its pinned host copy is complete
   before the CUDA page is eligible for eviction.
3. Prefetch records a CUDA event; a graph or eager kernel may read the page only
   after that event completes on the consumer stream.
4. Eviction cannot race a graph replay, verify snapshot, replay, cancellation,
   or page-table update. Logical token positions and page IDs never change.
5. GDN, QSA/indexer, PLE, replay and speculative state remain CUDA-resident or
   receive the same explicit ownership/version protocol; implicit UVM is not
   allowed.
6. OOM, timeout, cancellation and restart release host pins and CUDA events
   idempotently. Eager mode must remain a complete fallback.

## Failure matrix

| Failure | Required response | Current support |
|---|---|---|
| Host allocation/pin failure | Keep all-VRAM path or fail before mutation | Not implemented |
| Prefetch event late | Do not launch dependent graph; use eager/wait path | Not implemented |
| Duplicate/late eviction event | Ignore by page version; retain owner | Not implemented |
| Graph capture/rebind mismatch | Disable tiering and use eager/all-VRAM | Capture fallback exists; tier fallback absent |
| Speculative reject/rollback | Restore page, QSA, PLE and GDN versions atomically | CUDA-only snapshots exist |
| Cancellation/restart | Return pages, slots, pins and events exactly once | CUDA-only lifecycle exists |
| OOM during transition | Preserve old pool and report margin | Rebuild rejection exists; host tier absent |

## Minimal implementation plan if re-opened

1. Add a page-owner/version state machine independent of model geometry.
2. Add pinned host backing and bounded copy queues with CUDA events.
3. Add an explicit graph residency set; graph replay refuses cold pages.
4. Add eager fallback and an opt-out flag before automatic activation.
5. Prove unit invariants for ownership, ordering, rollback, cancellation and
   restart, then a CUDA round-trip/address test.
6. Run paired 4K/16K inactive-path tests before forcing one threshold-crossing
   context. Only then consider 64K–256K qualification.

## Decision

The current runtime keeps expert arenas allocated and allocates prefill
temporaries separately. A planner-only accounting change would claim memory that
is not physically reusable and can cause OOM or invalidate graph addresses.
Therefore KV-RAM tiering remains disabled; no long-context certification may
start until the allocator and CUDA integration proof above exist.

## Campaign 14 implementation (2026-09-25)

Branch `kvram-live` (worktree `../ft-kvram-live`): `faccdc8` live tier, `d7f22fa` hot floor + rebalancer, `8b094e9` auto spill + RAM budget, `e453a98` run history.

- **Placement:** highest physical page ids live in page-locked host RAM (`cudaHostRegister`; `pin_memory` rounds to powers of two). No migration protocol: the QSA attention kernel reads RAM pages in place (zero-copy UVA) for decode/verify/graph replay; eager prefill (>8 rows) gathers touched RAM pages into a device staging slab per layer. Stores route per slot. Everything is stream-ordered: no events, no per-token host sync, fixed graph addresses.
- **Always on device:** compressed index rows, rope rows, pending ring, GDN/PLE state, page table. Selection runs on device; only selected K/V pages cross PCIe.
- **Hot/cold:** scheduler page ids are logical; kernels address `page_map[logical]`. Decode selections accumulate per-page heat; every 16 steps the hottest RAM pages swap with the coldest device pages (K/V, index rows, rope rows) on the scheduling stream between forwards. Radix tree, page table and free list never observe a move.
- **Planner:** with a RAM tier the device keeps `kv_reserve_tokens` of KV; the rest funds expert slots.
- **Modes:** `auto` (default), `off`, `force` (`--kv-cache-ram`, `--kv-ram-tokens`, `--kv-reserve-tokens`). `auto` tiers cold KV into RAM by default for a *certified* family (measured faster from 64K) and drops the tier to all-VRAM when RAM cannot hold it; certification is a model-declared capability (`ModelConfig.kv_ram_tier_certified`, set by qwen4_exp today), not a name match, so uncertified families keep all-VRAM under `auto` until measured. Unsupported (non-QSA pool, uncertified family under `auto`, TP>1, non-CUDA): `force` fails closed, `auto` logs the reason and stays in VRAM.
- **RAM budget:** refuse before allocation when the tier would cross `MemAvailable - 10% MemTotal - 2 GiB` (earlyoom SIGTERMs at 10%), reporting the largest context the host can hold.

### Evidence (UD-IQ4_XS unless noted; cold, fresh server, 256 decode, exact prompts)

| Run | PP | TG | experts | hash |
|---|---|---|---|---|
| 4K off | 1464.9 | 47.30 | 3519 | 3af3056b98c0 |
| 4K force, 2K device | 1234.2 | 45.86 | 3500 | 3af3056b98c0 |
| 4K auto | 1463.5 | 47.24 | 3519 | 3af3056b98c0 |
| 16K off | 2051.0 | 45.11 | 2994 | 9aea3d2eda1e |
| 16K force, 8K device | 1989.8 | 44.56 | 2956 | 9aea3d2eda1e |
| 16K hot 2K (rebalancer) | 2055.3 | 43.47 | 3122 | 9aea3d2eda1e |
| 16K auto | 2141.0 | 45.01 | 2994 | 9aea3d2eda1e |
| UD-sq4 k1 16K off / hot 2K | 2147.8 / 2056.9 | 50.70 / 48.30 | 2786 / 2966 | 99e6c0f8c99f |
| ISTA 4K off / force 2K | 1648.2 / 1582.6 | 58.97 / 56.86 | 5555 / 5500 | 2e4a55acfa90 |

Kernel tests: `tests/kvcache/test_kv_ram_tier_cuda.py` (25, bit-exact vs all-VRAM incl. graph replay, staging, rebalance, fault injection). Logs: `/models/desenvolvimento/ft-campaign2/campaign14/`.

### Decisions and open items

- The auto cold tier narrows to fit RAM by a measured ladder (fp8 -> turbo4 -> turbo3; bf16/turbo8 dropped as dominated), so a certified family keeps KV in RAM at every context instead of only overflowing. ISTA 64/128/256K fp8 beats all-VRAM (see `PERFORMANCE.md`).
- Host RAM is bound by pinned experts (server RSS 50-70 GiB during load; earlyoom kills at <10% free): >~10 GiB of KV in RAM needs a compressed cold tier or NVMe-streamed cold experts.
- Not done: compressed cold tier (turbo3/turbo4 host tier now ships in the auto ladder), `ft bench context` advisor, VRAM-side max-context message (RAM-side exists). Certification of families other than qwen4_exp for the KV-in-RAM default is pending measurement (they stay all-VRAM until then).

## Campaign 18 phase 0: MoE 35B and dense 27B — NO-GO (2026-09-25)

Dense full attention reads the whole KV every decode step. Measured PCIe H2D 56 GB/s; fp8 KV per token 10 KiB (35B, 10 KV layers) / 32 KiB (27B, 16 KV layers).
At 128K that is 1.25 / 4.0 GiB of PCIe per token (24 / 77 ms), versus 11 ms (35B NVFP4) / ~40 ms (27B) for the whole all-VRAM step.
Estimated RAM-tier TG: 35B ~29 vs 89 tok/s @128K; 27B ~13 vs 25 @64K. Freed VRAM (2.5-5 GiB) cannot repay it. Both families keep KV in VRAM;
no code added. Math and alternatives: `ft-campaign2/campaign18/LEDGER.md`.
