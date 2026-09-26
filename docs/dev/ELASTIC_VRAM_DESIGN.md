# Elastic VRAM (Windows 8 GB and up) — design

Status: design; the Linux-testable bricks are implemented and GPU-verified (campaign 20).

## Goal (operator requirements)
- The user sets only the context. The engine never OOMs, always balances experts against context, and never
  grows VRAM past the startup plan.
- Windows/WDDM silently spills VRAM to system RAM when a process exceeds its budget (a slowdown, not an
  error). The engine must stay under the per-process budget so pressure is a catchable event, never a spill.

## Bricks already in `next`
| brick | where | verified |
|---|---|---|
| Startup plan prices every consumer; the expert cache is the one elastic consumer | `engine/memory_planner.py`, `engine/vram_ledger.py` | account always >= allocator-held (campaign 20, 24 runs) |
| Idle guard: shrink experts when the last busy window's peak left < 256 MiB free | `Engine.guard_vram_at_idle` | campaign 19 |
| Idle guard: regrow experts after 3 calm windows (> 2 x 256 MiB free at peak), capped at the startup plan, through the rebuild budget check | same | campaign 20: 7623 -> 7647 -> 8024 (plan) |
| OOM inside a forward fails that request (error reply), frees its pages, shrinks experts 5%, server keeps serving | `Scheduler._forward_or_fail` / `_flush_oom`, `Engine.shrink_after_oom` | campaign 20: real OOM from a second process on a live server, overlap + MTP paths |
| MTP is dropped before the context is refused or narrowed | `engine._shed_mtp`, `Engine._release_mtp` | campaign 20 |

## Windows-only pieces (to build)
1. **Budget source.** `IDXGIAdapter3::QueryVideoMemoryInfo(DXGI_MEMORY_SEGMENT_GROUP_LOCAL)` gives `Budget` and
   `CurrentUsage` for this process. The planner's baseline becomes `min(driver_free, Budget - CurrentUsage)`
   instead of `cudaMemGetInfo` alone (WDDM `free` does not reflect the budget).
2. **Event, not poll.** `RegisterVideoMemoryBudgetChangeNotificationEvent` signals a Win32 event on budget change.
   A small native thread (ctypes or the existing C++ extension) waits on it and sets a flag the scheduler reads
   at its idle safe point — the same point where `guard_vram_at_idle` runs. Linux keeps the poll
   (`mem_get_info` at idle).
3. **Reaction.** Budget down: shrink the expert cache by the shortfall + margin through the existing rebuild
   (pure shrink always fits). Budget up: the regrow path, still capped at the startup plan.
4. **Spill guard.** Document and check at startup: NVIDIA Control Panel -> Manage 3D settings -> Program
   settings -> python.exe -> "CUDA - Sysmem Fallback Policy" = "Prefer No Sysmem Fallback". With it, pressure
   becomes a CUDA OOM (handled by `_forward_or_fail`) instead of a silent 5-10x slowdown. The engine cannot set
   it; it logs a warning when `CurrentUsage > Budget` is observed without an OOM.
5. **Bounded runtime caches.** Every runtime cache must be priced at startup and bounded (prefix/GDN snapshot
   caches, graph pools, spec states). Campaign 20 account check: never under-priced; over-priced 0.24-0.6 GiB
   (graph pool is a formula, not measured — next step: reserved-memory delta across capture).

## Test plan
- Linux (done): hog VRAM from a second process on a live server -> error reply, re-serve, regrow to plan.
- Windows 8 GB card: start at the largest context the planner accepts; open a game/browser to cut the budget;
  expect a budget event -> shrink, no spill (Task Manager "Shared GPU memory" flat), TG recovers after regrow.
