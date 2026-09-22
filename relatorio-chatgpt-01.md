# FreeToken-Next VRAM Auto-Planning Investigation

**Repository:** `/models/desenvolvimento/freetoken-next`  
**Branch:** `next`  
**Inspected commit:** `20ca91764feab5ed244775546c2caf74bfacebd8`  
**Scope:** diagnostic only; no production code, configuration, tests, benchmarks, or existing logs were modified.

## 1. Executive finding

The current planner can produce an internally valid byte plan and still OOM during a real 16K prefill because the plan commits the entire negotiable pool to persistent expert/KV allocations before the first real GDN prefill. The plan's transient reserve is a static/modelled estimate, while the actual first GDN forward allocates a tensor that is not available as free physical VRAM after those pools are committed.

The latest real 16K evidence is:

- Model: `/models/Qwen3.8-Flash-Next-NVFP4-Radix`.
- KV format/backend: `turbo4`, resolved to `qsa_sparse`.
- Requested context: 16,384 tokens.
- Prompt: 15,800 tokens.
- MTP: disabled (`spec_mtp=0`).
- CUDA graphs: disabled (`cuda_graph_max_bs=0`).
- MoE: `offload`, automatic expert cache enabled.
- Initial free memory: 15.18 GiB.
- Planner: 1,262 expert slots and 259 usable KV pages, 16,576 tokens.
- Free memory after initialization: 1.24 GiB.
- The server became ready and accepted the request.
- The first real prefill failed in GDN/FLA while allocating `h`.
- Failed allocation: 192.00 MiB.
- Driver-reported free memory at failure: 171.38 MiB.
- Process memory in use: 15.20 GiB, of which PyTorch allocated 14.73 GiB and reserved-but-unallocated 84.28 MiB.
- The scheduler worker exited; no complete generation or JSON benchmark result was produced.

The exact failing path is:

`engine.forward_batch` -> `qwen4_exp.model.forward` -> `qwen4_exp.gdn.forward` -> `gdn_prefill_chunk_fla` -> `chunk_gated_delta_rule` -> `chunk_gated_delta_rule_fwd_h` -> `h = k.new_empty(B, NT, H, V, K)`.

This is not a hypothetical OOM and not merely a benchmark watchdog problem. The first real prefill reached the model and failed at a concrete allocation.

## 2. Current worktree and evidence status

The worktree is not clean. Relevant modified files include:

- `python/freetoken/engine/config.py`
- `python/freetoken/engine/engine.py`
- `python/freetoken/moe/offload_cache.py`
- `python/freetoken/server/args.py`
- `benchmarks/bench_pp_tg.py`
- `python/freetoken/scheduler/scheduler.py`
- `python/freetoken/scheduler/spec.py`
- `python/freetoken/models/qwen4_exp/model.py`
- `python/freetoken/models/qwen4_exp/gguf.py`
- `python/freetoken/models/qwen3_5_moe/gguf_experts.py`
- `python/freetoken/models/gguf/reader.py`
- `scripts/bench-mtp-equiv.sh`
- `scripts/wait-for-server.sh`
- tests related to GGUF/offload.

Relevant untracked files include:

- `scripts/test-runner.py`
- `tests/test_runner.py`
- `benchmarks/results/campaign-20260920/`
- `relatorio-calculo-automatico.txt`
- `goal-20260920-2037.txt`.

The source inspected is therefore the current worktree, not necessarily the committed baseline. The report must not attribute every current behavior to the inspected commit.

The newest relevant real GPU execution by filesystem modification time is:

`benchmarks/results/campaign-20260920/auto-context-16k-k0.log`

Its timestamp is newer than the other campaign logs inspected. Later attempted runs did not produce a valid JSON result and were not treated as successful GPU evidence.

## 3. Relevant source inventory

### 3.1 Configuration and CLI

#### `python/freetoken/engine/config.py`

`EngineConfig` contains the primary sizing inputs:

- `kv_format: str = "auto"`
- `moe_cache_size: int = 0`
- `moe_cache_auto: bool = False`
- `kv_reserve_tokens: int = 8192`
- `kv_reserve_context: bool = True` in the current worktree
- `moe_prefill_overlap: bool = True`
- `cuda_graph_bs`
- `cuda_graph_max_bs`
- `memory_ratio: float = 1.0` in the current worktree
- `max_seq_len_override`
- `num_page_override`
- `num_token_override`
- `max_extend_tokens` is inherited/added by `SchedulerConfig`.

`max_seq_len` returns `max_seq_len_override` when present, otherwise the model rotary maximum. `max_forward_len` in `SchedulerConfig` returns `max_extend_tokens`, not the total serving context.

#### `python/freetoken/scheduler/config.py:14-39`

`SchedulerConfig.max_extend_tokens` defaults to `8192`. Its `max_forward_len` property returns this value. This creates two distinct dimensions:

- total KV serving capacity: `max_seq_len`;
- one scheduler forward/prefill budget: `max_extend_tokens`.

#### `python/freetoken/server/args.py`

The CLI still exposes `--memory-ratio`. Although the current default is 1.0 and several automatic paths force 1.0, the field and option remain part of the operational model.

The CLI also exposes `--max-extend-tokens`, `--max-seq-len-override`, KV format, MoE strategy, MoE cache auto, graph settings, and explicit page/token overrides.

### 3.2 Memory measurement

#### `python/freetoken/engine/graph.py:101-103`

`get_free_memory(device)` returns:

```python
torch.cuda.mem_get_info(device)[0]
```

This is driver-visible physical free memory, not PyTorch allocated bytes.

#### `python/freetoken/engine/engine.py:1150-1167`

`Engine._sync_get_memory()`:

1. synchronizes the device;
2. calls `torch.cuda.empty_cache()`;
3. calls `torch.cuda.reset_peak_memory_stats()`;
4. reads `get_free_memory()`;
5. all-reduces free memory across TP ranks;
6. returns minimum and maximum free memory.

It uses `mem_get_info`, not `memory_reserved`, for startup baselines.

#### `python/freetoken/engine/engine.py:390-397`

After loading the model, the engine computes:

```text
_weights_bytes = baseline_free - post_weights_free
```

This difference is treated as resident model weight cost. It is a physical free-memory difference, not a sum of model parameter tensors.

#### `python/freetoken/engine/engine.py:511`

After KV/state/page-table setup, the engine stores:

```python
_transient_probe_base = int(torch.cuda.memory_allocated(self.device))
```

#### `python/freetoken/engine/engine.py:765-783`

`_calibrate_vram_ledger()` reads:

- `torch.cuda.memory_allocated()` as `measured:allocator-held`;
- `torch.cuda.max_memory_allocated() - _transient_probe_base` as `measured:transient-peak`.

The current engine does not use `torch.cuda.memory_reserved()` or `torch.cuda.max_memory_reserved()` in this calibration. The scheduler has a separate `memory_reserved()` accessor at `python/freetoken/scheduler/scheduler.py:528`, but it is not the startup planner's accounting source.

There is no allocator snapshot analysis in the current planner. There is no complete accounting of driver allocations that are neither PyTorch allocated nor PyTorch reserved.

### 3.3 Ledger and charges

#### `python/freetoken/engine/vram_ledger.py`

`VramLedger` stores named `Charge` entries with a byte count and `Kind`:

- `IMMUTABLE`
- `PERSISTENT`
- `SEMI_PERSISTENT`
- `TRANSIENT`
- `RESERVE`
- `MEASURED`

The negotiable consumers are defined as:

```python
NEGOTIABLE = ("cache:expert", "cache:kv")
```

Measured lines are intentionally excluded from normal totals so the diagnostic measurement is not charged a second time as a consumer.

Important methods:

- `charge()` replaces an existing named charge rather than adding it;
- `total()` sums selected charge kinds and excludes measured lines by default;
- `reserve_bytes` returns `max(modelled_reserve_bytes, measured_transient_peak)`;
- `engine_committed_bytes()` excludes modelled reserve lines and measured lines;
- `pool_budget_bytes()` returns the ceiling minus committed non-negotiable consumers;
- `decide()` performs the MoE-first split;
- `plan_for_context()` attempts to fund requested context first but ultimately calls `decide()` with the requested context as a KV floor.

### 3.4 Budget policy

#### `python/freetoken/engine/cache_budget.py:45-75`

`ceiling_bytes()` is:

```text
cap = int(memory_ratio * baseline_free)
implicit_reserve = baseline_free - cap
ceiling = cap - max(0, reserve_bytes - implicit_reserve)
```

`net_cache_budget_bytes()` is:

```text
net_budget = ceiling_bytes(
    baseline_free,
    memory_ratio,
    reserve_bytes,
) - weights_bytes - fixed_cache_size
```

The current code still carries the ratio-based policy even though the automatic engine path uses 1.0 in important places.

#### `python/freetoken/engine/vram_ledger.py:697-710`

The ledger's pool budget is:

```text
committed = engine_committed_bytes() + extra_fixed_bytes
pool_budget = max(0, ceiling_bytes - committed)
```

`extra_fixed_bytes` is intended for the KV pool's fixed tier.

#### `python/freetoken/engine/cache_budget.py:78-95`

The KV pool allocates a dummy page in addition to usable pages:

```text
pool_pages(num_pages) = num_pages + 1
required_bytes =
    moe_cache_size * per_expert_bytes
    + (num_pages + 1) * cache_per_page
```

#### `python/freetoken/engine/cache_budget.py:98-161`

`plan_cache_budget()` performs the split:

1. `hi = min(total_experts, max_slots)`.
2. `overlap = prefill_overlap and hi >= 2 * num_experts`.
3. `lo = 2 * num_experts` if overlap, otherwise `num_experts`.
4. Reserve KV floor including dummy page:

```text
kv_reserve_bytes = (kv_reserve_pages + 1) * cache_per_page
```

5. Greedy expert fill:

```text
raw = (budget_bytes - kv_reserve_bytes) // per_expert_bytes
moe_cache_size = max(lo, min(raw, hi))
```

6. Remaining pages:

```text
remaining = budget_bytes - moe_cache_size * per_expert_bytes
num_pages = max(
    remaining // cache_per_page - 1,
    kv_reserve_pages,
)
```

7. It computes `required_bytes()` and gives back expert slots if the page-floor rounding makes the plan exceed budget.
8. It asserts the final plan fits and has more than one page.

This is mathematically consistent for the quantities passed to it. The problem is that the passed budget and per-consumer quantities do not fully represent all memory live during the real prefill.

### 3.5 Automatic MoE sizing

#### `python/freetoken/engine/engine.py:830-885`

`Engine._resolve_auto_moe_cache_size()` obtains:

- `cache_per_page`
- fixed KV cost
- `page_tokens`
- minimum reserve
- `num_experts`
- `total_experts = num_moe_layers * num_experts`
- `per_expert_bytes = expert_bytes_per_slot(banks.sources)`
- prefill overlap
- method slot limit.

When `kv_reserve_context` is true or `max_seq_len_override` is set, it calls:

```python
plan = self.vram_ledger.plan_for_context(config.max_seq_len, **geometry)
```

Otherwise it calls `decide()` with:

```text
kv_reserve_tokens = max(
    config.kv_reserve_tokens,
    min_reserve,
    requested_tokens_from_page_override,
)
```

The returned plan writes `config.moe_cache_size` and, when no explicit page override exists, writes `config.num_page_override`.

#### `python/freetoken/engine/cache_budget.py:17-42`

`expert_bytes_per_slot()` is geometry-based. It iterates all source banks, deduplicates `(shape[1:], dtype)` within each bank, and sums one row's bytes for each distinct geometry.

This function does not include cache bookkeeping tensors, cache copy-plan tensors, prefill buffers, expert auxiliary tensors, or all other GPU tensors reachable from the cache object. Those are handled later or not at all.

### 3.6 KV geometry

#### `python/freetoken/kvcache/base.py`

`BaseKVCache.kv_cost()` is abstract. Each KV family returns:

```text
(cache_per_page, fixed_cache_size, page_tokens, min_reserve_tokens)
```

The family owns only its own buffers. The engine separately adds GDN state and other sibling costs.

#### `python/freetoken/kvcache/qsa_pool.py:217-240`

The QSA pool's `kv_cost()` combines the full-attention pool's cost and QSA-specific fixed/slab costs. `unit_bytes()` reports runtime unit costs.

#### `python/freetoken/kvcache/turbo_pool.py:37-43,258-275`

Turbo storage calculates packed bytes per token from:

- head dimension;
- KV head count;
- code bytes for Turbo3/Turbo4;
- norm bytes;
- number of slabs;
- number of model layers.

Turbo3 and Turbo4 therefore have distinct byte costs derived from geometry, rather than from a user memory percentage.

### 3.7 GDN and FLA

#### `python/freetoken/engine/vram_ledger.py:264-296`

`gdn_prefill_bytes()` estimates one GDN layer's prefill workspace from model dimensions and `tokens`. It includes terms for:

- `conv_in`;
- `z`;
- q/k/v reshape copies;
- `w` and `u`;
- `A`;
- `h`;
- `v_new`;
- output `o`;
- fp32 gate/beta.

It uses fixed `GDN_CHUNK_SIZE = 64` for the per-chunk state term. It assumes the model's layers execute sequentially, so this is a per-layer peak, not a sum over layers.

#### `python/freetoken/models/qwen3_5_moe/gdn_kernels.py:6-48`

`gdn_prefill_chunk_fla()` calls `chunk_gated_delta_rule()` and returns the output. The actual `h` allocation occurs deeper in the FLA kernel.

#### `python/freetoken/kernel/fla/chunk_delta_h.py:278` in the OOM trace

The failed allocation is:

```python
h = k.new_empty(B, NT, H, V, K)
```

The exact log stack reports the requested size as 192.00 MiB. The exact shape/dtype values at failure are not printed in the current log, so the report cannot derive the exact element count from the log alone. The source shows that its size scales with:

- batch `B`;
- number of chunks `NT`;
- value heads `H`;
- value dimension `V`;
- key dimension `K`;
- dtype inherited from `k`.

`NT` is determined by the prefill token count and FLA chunking. It therefore scales with the actual prefill chunk and can be reduced by reducing the scheduler prefill chunk, while total KV context can remain 16K.

#### `python/freetoken/kernel/fla/chunk_o.py:128-136`

The output path computes:

```python
B, T, Hg, K, V = *q.shape, v.shape[-1]
H = v.shape[-2]
BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
NT = triton.cdiv(T, BT)  # when cu_seqlens is None
 o = torch.zeros_like(v)
```

The OOM trace is specifically in `chunk_delta_h.py`, not `chunk_o.py`; earlier descriptions that named `torch.zeros_like(v)` as the failed allocation are contradicted by the newest log. The current newest trace identifies `h = k.new_empty(...)` as the failed allocation.

The estimator's `h` term is based on one `[V,K]` state per 64-token FLA chunk, but the exact actual `NT`, dtype, tensor lifetime, and all simultaneously live inputs are not logged. This prevents a complete byte-for-byte reconciliation from source and current log alone.

### 3.8 Prefill chunk selection

`SchedulerConfig.max_extend_tokens` defaults to 8192. `Scheduler.__init__()` sets:

```text
prefill_budget = min(max_extend_tokens, cache_manager.prefill_chunk_budget)
```

when a cache-specific cap exists; otherwise it is `max_extend_tokens`.

`PrefillManager._add_one_req()` starts with:

```text
remain_len = input_len - cached_len
chunk_size = min(token_budget, remain_len)
```

It may further reduce the chunk for sliding-window page availability and alignments. For the Qwen Flash Next case in the latest log, `max_extend_tokens=8192` and the GDN ledger line explicitly describes an 8192-token forward.

Total context and prefill chunk are separate in the scheduler. The current planner does not solve `max_extend_tokens` from the physical memory remaining after final persistent cache allocation. It models the configured chunk before the real forward, but does not feed a solved chunk back into the scheduler.

### 3.9 CUDA Graphs

#### `python/freetoken/engine/graph.py:105-199`

`GraphRunner` derives graph batch sizes, initializes the attention backend for capture, synchronizes, empties the PyTorch cache, resets peak stats, allocates graph buffers, and captures graphs for each batch size.

In the latest failing run, `cuda_graph_max_bs=0`, so graph capture was disabled and the log says `CUDA graph is disabled.` Therefore CUDA Graph allocations are not part of the 16K failure's active path, although the planner still supports graph-related reserve calculations for other configurations.

For graph-enabled configurations, graph capture happens after the automatic MoE/KV plan has already been committed. Graph pool memory can therefore be a late consumer unless its estimate is exact.

### 3.10 Triton/autotune and attention workspaces

`modelled_reserves()` adds:

- `TRITON_AUTOTUNE_ARENA`;
- graph capture peak;
- `BACKEND_WORKSPACE`;
- graph pool;
- activation peak;
- GDN prefill peak;
- fragmentation reserve.

In the current worktree these include host-calibrated constants in `python/freetoken/engine/vram_ledger.py:45-89`, including:

- `TRITON_AUTOTUNE_ARENA = 128 * MiB`;
- `GRAPH_POOL_FIRST_SHAPE = 200 * MiB`;
- `GRAPH_POOL_EXTRA_SHAPE = 150 * MiB`;
- `GRAPH_CAPTURE_PEAK = 128 * MiB`;
- `BACKEND_WORKSPACE = 128 * MiB`;
- `MM_ENCODER_PEAK = 192 * MiB`;
- `FRAGMENTATION_RESERVE = 64 * MiB`;
- `GRAPH_CAPTURE_EXTRA_SHAPE = 160 * MiB`;
- `LIVE_ACTIVATION_TENSORS = 6`.

These are not user flags, but they are heuristic or host-calibrated policy inputs. Some are not active in the latest run: graphs are disabled and multimodal encoders are disabled. Triton autotune and attention workspace remain active.

## 4. Complete initialization timeline

The following timeline is reconstructed from `Engine` and the latest log.

### Stage 1: process and configuration

- CLI resolves `kv_format=turbo4`, `moe_strategy=offload`, `moe_cache_auto=True`, `max_seq_len_override=16384`, `max_extend_tokens=8192`, `cuda_graph_max_bs=0`.
- No GPU model/cache bytes have yet been committed by the engine planner.
- The total serving context and one-forward chunk are already separate values.

### Stage 2: initial memory measurement

- `_sync_get_memory()` synchronizes, empties PyTorch cache, resets peak stats, and reads driver free memory via `mem_get_info`.
- Latest log: 15.18 GiB free.
- This is physical driver free memory at the measurement moment.

### Stage 3: model creation and weight loading

- Model modules are created on meta device.
- State dict is loaded and quantization finalized.
- Model weights and any model-resident GPU tensors are allocated.
- The engine measures post-weight free memory.
- `_weights_bytes` is computed as the free-memory difference.
- The latest log does not print the explicit post-weight measurement before expert loading, but the ledger later charges `weights:model = 9.459 GiB`.

### Stage 4: host tables/PLE

- Host-side PLE tables are loaded if applicable.
- The latest deployment uses disk PLE and reports zero GPU PLE bytes. Host pinned bytes are not GPU VRAM but affect system memory/pinning, not the GPU ledger.

### Stage 5: ledger creation

- `_open_vram_ledger()` reads total device memory.
- It chooses `prefill_tokens = config.max_extend_tokens`, therefore 8192.
- It calls `modelled_reserves()` with model geometry, hidden size, dtype, graph settings, backend workspace enabled, and encoder state.
- It charges weights, modelled transient/semi-persistent/reserve lines, GDN state pool, PLE GPU line, and page-table estimate.
- At this point expert and KV pools do not yet exist.

### Stage 6: automatic expert/KV split

- Expert banks are loaded/constructed.
- Auxiliary alpha bytes are charged when present.
- `_resolve_auto_moe_cache_size()` calls `plan_for_context(16384, geometry)`.
- The pool budget is 3.369 GiB.
- The split commits 3.259 GiB to 1,262 expert slots and 0.110 GiB to 259 usable KV pages.
- `config.moe_cache_size` and `config.num_page_override` are set.

The critical commitment occurs here: the expert cache and KV page count are selected before the first real prefill and leave effectively zero ledger headroom.

### Stage 7: expert cache allocation

- `OffloadMoeCache` allocates GPU bank caches and bookkeeping tensors.
- Bank sources remain host/pinned or lazy sources depending on format.
- Auxiliary expert tensors/scales may be materialized.
- The current `_charge_expert_cache()` measures reachable CUDA tensors after allocation and re-prices the ledger, but this is after the plan was chosen.
- In the latest log, the side-table charge prints 0.000 GiB despite a note naming `bank_caches.gate_up=1.926`, which indicates that the diagnostic note alone cannot be treated as a byte charge. This is an unresolved accounting contradiction requiring targeted measurement.

### Stage 8: KV pool allocation

- The engine creates the KV pool with the planned number of usable pages.
- The pool allocates usable pages plus a dummy page and any fixed/scratch structures.
- The pool's actual allocation can differ from the pre-allocation `kv_cost()` estimate if auxiliary structures or allocator behavior are not represented in that method.

### Stage 9: page table and state setup

- Linear/GDN state pools, page table, and token/page bookkeeping are allocated.
- The page-table ledger line is re-priced after the actual page table exists.
- The latest log shows `cache:gdn-state = 0.969 GiB` and page-table charge rounded to 0.000 GiB.

### Stage 10: attention backend and sampler

- The attention backend is created.
- Backend workspace may be allocated lazily or during initialization/warmup. The ledger uses a fixed `BACKEND_WORKSPACE` estimate.

### Stage 11: transient probe base

- `_transient_probe_base` is recorded using `torch.cuda.memory_allocated()`.
- This occurs after persistent pools and initialization structures already exist, not before the automatic cache split.

### Stage 12: graph/autotune initialization

- With graphs enabled, graph buffers and capture allocations occur after the plan.
- In the latest run graphs are disabled.
- Prefill autotune/warmup allocations occur after the plan and are represented by a fixed modelled arena plus later peak diagnostics.

### Stage 13: calibration

- `_calibrate_vram_ledger()` measures `memory_allocated` and `max_memory_allocated`.
- It does not automatically resize the already committed pools.
- It does not persist a profile.
- It warns on discrepancies rather than preventing the first real request.

### Stage 14: first real prefill

- Request is accepted after server readiness.
- The prefill scheduler sends an 8192-token chunk.
- GDN/FLA attempts `h = k.new_empty(B, NT, H, V, K)`.
- The physical driver cannot satisfy 192 MiB with only 171.38 MiB free.
- Scheduler worker exits.

## 5. Exact 16K failure evidence

Source log: `benchmarks/results/campaign-20260920/auto-context-16k-k0.log`.

Command recorded in line 1:

```text
/models/desenvolvimento/freetoken-next/.venv/bin/python -m freetoken.cli serve \
  --model /models/Qwen3.8-Flash-Next-NVFP4-Radix \
  --host 127.0.0.1 --port 49611 \
  --max-running-requests 1 \
  --max-seq-len-override 15896 \
  --cuda-graph-max-bs 0 \
  --max-seq-len-override=16384 \
  --kv-format=turbo4 \
  --moe-strategy=offload \
  --moe-cache-auto \
  --cuda-graph-max-bs=0 \
  --text-model-only \
  --disable-moe-prefill-overlap
```

The duplicate `--max-seq-len-override` and duplicate graph option come from the benchmark command construction. The resolved config uses 16,384 and graph max batch size 0.

Important log fragments:

```text
Free memory before loading model: 15.18 GiB
```

```text
Resolved config: moe_strategy='offload', attention_backend='qsa_sparse', cache_type='hybrid_radix', page_size=64
```

```text
memory plan: pool budget 3.369 GiB -> 1262 expert slots (3.259 GiB) + 259 usable KV pages (16576 tokens at 64/page) of the 0.110 GiB left for KV, prefill_overlap=False
```

```text
Allocating 16576 tokens for KV cache, K + V = 0.11 GiB
Free memory after initialization: 1.24 GiB
CUDA graph is disabled.
```

Ledger lines:

```text
weights:model                9.459 GiB
cache:expert                 3.259 GiB
cache:gdn-state              0.969 GiB
cache:kv                     0.110 GiB
workspace:attention          0.125 GiB
transient:gdn-prefill        0.835 GiB
transient:activations        0.234 GiB
transient:autotune           0.125 GiB
reserve:fragmentation        0.062 GiB
measured:allocator-held     13.659 GiB
```

The report then says:

```text
committed 15.179 GiB, ceiling 13.922 GiB
headroom -0.000 GiB under the ceiling, uncommitted -0.000 GiB
```

It also says:

```text
VRAM ledger over-modelled the account by 0.26 GiB: the pools it priced are smaller than the account claimed, so context is being left unspent.
```

The request is accepted:

```text
model_id=Qwen3.8-Flash-Next-NVFP4-Radix ctx=16384 attn=hybrid_linear moe=True
POST /v1/completions HTTP/1.1 200 OK
```

The scheduler then fails:

```text
File .../python/freetoken/kernel/fla/chunk_delta_h.py, line 293, in chunk_gated_delta_rule_fwd_h
    h = k.new_empty(B, NT, H, V, K)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 192.00 MiB. GPU 0 has a total capacity of 15.51 GiB of which 171.38 MiB is free. Including non-PyTorch memory, this process has 15.20 GiB memory in use. Of the allocated memory 14.73 GiB is allocated by PyTorch, and 84.28 MiB is reserved by PyTorch but unallocated.
```

The backend worker exits with code 1. The API supervisor later stops the server. No output SHA1 or complete generation exists for this run.

## 6. Predicted versus actual reconciliation

The following table uses exact values printed in the latest log where available. Values are rounded by the log to GiB.

| Quantity | Value | Evidence/meaning |
|---|---:|---|
| Initial driver free memory | 15.18 GiB | `mem_get_info` before model loading |
| Device total | 15.51 GiB | CUDA OOM message |
| Model weight charge | 9.459 GiB | Ledger immutable line |
| Expert cache planned | 3.259 GiB | 1,262 × 2.64 MiB |
| GDN state pool | 0.969 GiB | 9 physical slots |
| KV cache planned | 0.110 GiB | 259 usable + dummy page |
| Attention workspace estimate | 0.125 GiB | Semi-persistent fixed estimate |
| GDN transient estimate | 0.835 GiB | 8,192-token chunk formula |
| Activation estimate | 0.234 GiB | 6 tensors × hidden/token estimate |
| Triton autotune estimate | 0.125 GiB | Fixed arena estimate |
| Fragmentation reserve | 0.062 GiB | Fixed reserve |
| Allocator-held after init | 13.659 GiB | `torch.cuda.memory_allocated` |
| Ledger committed display | 15.179 GiB | Ledger report includes its committed calculation, not the measured line |
| Ledger ceiling | 13.922 GiB | baseline 15.179 GiB, ratio 1.0, reserve 1.257 GiB |
| Free after initialization | 1.24 GiB | Driver free memory |
| PyTorch allocated at OOM | 14.73 GiB | OOM message |
| PyTorch reserved but unallocated at OOM | 84.28 MiB | OOM message |
| Non-PyTorch-inclusive process use | 15.20 GiB | OOM message |
| Driver free at OOM | 171.38 MiB | OOM message |
| Failed allocation | 192.00 MiB | OOM message |

The values do not form one directly additive accounting domain:

- `15.20 GiB process in use` includes non-PyTorch allocations.
- `14.73 GiB allocated by PyTorch` excludes PyTorch reserved-but-unallocated blocks.
- `84.28 MiB reserved but unallocated` is allocator-reserved capacity, not necessarily immediately usable for the requested contiguous allocation under the allocator state.
- `171.38 MiB free` is driver-visible free physical VRAM.
- The ledger's `committed` number excludes or handles reserve/measured lines according to its own charge-kind rules and is not identical to `memory_allocated` or `mem_get_info`.

Definite facts:

- The failed allocation needed 192 MiB.
- Only 171.38 MiB physical free was available.
- The request therefore failed even before considering fragmentation: the requested block exceeded the reported free bytes by 20.62 MiB.
- The first real GDN prefill allocation was not included in the already committed physical pool budget at the time it was needed.

Strongly supported contributors:

- The planner filled the negotiable budget to effectively zero headroom.
- The GDN estimator was not sufficient to guarantee the real first prefill because the actual `h` allocation failed.
- The planner uses `memory_allocated` for calibration but `mem_get_info` for physical free memory and does not maintain a single allocator/driver-domain equation.
- The final first-forward transient may differ from the startup model due to Triton/JIT/backend behavior and tensor coexistence.

Unknown from available evidence:

- Exact shape and dtype of `h` at failure.
- Exact bytes of every simultaneously live GDN input and output tensor at the failing moment.
- Exact PyTorch reserved bytes immediately before the failed allocation beyond the OOM summary.
- Exact non-PyTorch allocations between the ledger report and the OOM.
- Whether the 20.62 MiB minimum shortfall alone explains all discrepancy or whether allocator fragmentation/non-PyTorch use materially increased the failure risk.
- Exact size of all expert side tables, because the log charge is rounded to 0.000 GiB and the diagnostic note names tensors without printing their total byte sum.

## 7. Analysis of the 192 MiB allocation

The newest OOM stack identifies the allocation exactly at source level as:

```python
h = k.new_empty(B, NT, H, V, K)
```

in `python/freetoken/kernel/fla/chunk_delta_h.py:293`.

Facts:

- `h` inherits device, dtype, and element type from `k`.
- Its size is `B × NT × H × V × K × element_size(k)`.
- `NT` is the number of FLA chunks for the current prefill segment.
- The current request is scheduled with an 8192-token forward budget.
- The failure occurs inside the first real prefill, after model, expert cache, KV cache, GDN state, attention setup, and autotune-related setup have already allocated memory.
- The source-level estimator includes an `h` term based on one `[V,K]` state per 64-token chunk, but the current evidence does not prove that its formula exactly matches `NT`, dtype, padding, packed-sequence behavior, or the simultaneous lifetime of every other tensor.

The exact shape cannot be recovered from the current log because the kernel does not log `B`, `NT`, `H`, `V`, `K`, or dtype at failure. A targeted instrumented GPU run would be required to print these values without changing the production behavior, or a source-level derivation would need all runtime shapes from the model config and batch.

The 192 MiB request is therefore identified by allocation site but not fully reconciled by shape.

## 8. GDN estimator audit

The estimator claims to price the following live terms:

- projection/`conv_in`;
- `z`;
- q/k/v reshape copies;
- `w`;
- `u`;
- `A`;
- `h`;
- `v_new`;
- `o`;
- gate/beta fp32.

The actual GDN path includes:

- model-side convolution/projection output preparation;
- q/k/v/g/beta tensors passed into `gdn_prefill_chunk_fla()`;
- the FLA autograd/function wrapper;
- `chunk_gated_delta_rule_fwd()`;
- `chunk_gated_delta_rule_fwd_h()` allocation of `h`;
- kernel output and hidden/intermediate tensors;
- post-kernel reshape and normalization.

The estimator is a symbolic peak approximation, not a runtime tensor liveness trace. The source/comments state that some tensors are counted as simultaneously live, but the report has no evidence that the following are exact:

- whether every listed tensor has the same dtype used by the implementation;
- whether the `h` term uses the exact runtime `NT` after padding/packed-sequence handling;
- whether output `o`, `v_new`, `h`, and input intermediates coexist at their maximum sizes;
- whether Triton kernel workspaces or compilation buffers coexist with the Python-visible tensors;
- whether the allocator rounds or reserves larger blocks than tensor `nbytes`.

The latest OOM disproves the proposition that the current estimator plus current pool plan is a sufficient upper bound for the real first prefill. It does not, by itself, distinguish an incorrect tensor-size formula from an incorrect lifetime model or an unmodelled allocator/backend allocation.

## 9. CUDA allocator semantics and accounting mismatch

The current system mixes at least three domains:

1. Driver physical free memory from `torch.cuda.mem_get_info()[0]`.
2. PyTorch currently allocated tensor bytes from `torch.cuda.memory_allocated()`.
3. PyTorch allocator reserved-but-unallocated bytes reported in the OOM message and accessible through `torch.cuda.memory_reserved()`.

The planner's startup baseline is domain 1. The model weight cost is inferred from a domain-1 free-memory difference. The post-init calibration is domain 2. The OOM includes domain 2, PyTorch reserved blocks, and non-PyTorch process usage.

The current engine does not read `memory_reserved()` or `max_memory_reserved()` in the startup ledger. Consequently:

- allocator-reserved blocks are not separately charged as a physical committed consumer;
- driver-visible non-PyTorch allocations are only reflected indirectly in lower `mem_get_info` values;
- tensor-walk accounting cannot see Triton/CUDA driver allocations without explicit measurement;
- `memory_allocated` cannot explain all physical in-use bytes.

`torch.cuda.empty_cache()` is called before some measurements. This releases eligible unused PyTorch cached blocks, but it does not destroy live tensors, does not release all driver allocations, and does not guarantee that a future contiguous allocation of a requested size will succeed.

`torch.cuda.reset_peak_memory_stats()` resets PyTorch allocation peak counters. It does not reset driver allocations or provide a complete physical-memory history.

The OOM itself explicitly reports a 15.20 GiB process use versus only 14.73 GiB PyTorch allocated, establishing that non-PyTorch/reserved domains account for a material difference.

## 10. Heuristics, constants, duplicated policy, and potential double counting

### Explicit constants in `vram_ledger.py:41-89`

- `GDN_CHUNK_SIZE = 64`: kernel chunk granularity; geometry-derived from the kernel convention, but hard-coded in the ledger.
- `TRITON_AUTOTUNE_ARENA = 128 MiB`: host-calibrated estimate.
- `GRAPH_POOL_FIRST_SHAPE = 200 MiB`: host-calibrated estimate.
- `GRAPH_POOL_EXTRA_SHAPE = 150 MiB`: host-calibrated estimate.
- `GRAPH_CAPTURE_PEAK = 128 MiB`: host-calibrated estimate.
- `BACKEND_WORKSPACE = 128 MiB`: host-calibrated estimate.
- `MM_ENCODER_PEAK = 192 MiB`: fixed per-image estimate.
- `FRAGMENTATION_RESERVE = 64 MiB`: explicit fixed reserve.
- `GRAPH_CAPTURE_EXTRA_SHAPE = 160 MiB`: host-calibrated estimate.
- `LIVE_ACTIVATION_TENSORS = 6`: liveness assumption.

The comments describe these as measured/calibrated on an RTX 5080/SM120, but the current source does not show a persistent calibration database or runtime keying by GPU/model/backend.

### Other hard-coded or policy values

- `SchedulerConfig.max_extend_tokens = 8192`.
- `EngineConfig.kv_reserve_tokens = 8192`.
- `DUMMY_PAGES = 1`.
- `plan_cache_budget()` minimum expert floor: `num_experts` or `2 * num_experts` when overlap is enabled.
- `total_experts = num_moe_layers * num_experts`.
- `page_size` is overridden to 64 for qsa_sparse in the latest run.
- `memory_ratio` remains in the formula and CLI despite the desired automatic policy.

### Possible missing/double counting

- The planner charges modelled transient lines as headroom and subtracts persistent/non-negotiable lines from the pool budget. This is intentional, but the exact distinction between `engine_committed_bytes()`, `reserve_bytes`, and the report's `committed` display must be traced carefully when comparing to physical memory.
- `cache:expert` is initially planned from `expert_bytes_per_slot`, then re-priced from `tensor_bytes(cache)` after allocation. This is too late to protect the initial allocation.
- Expert auxiliary bytes are charged before automatic planning in current worktree paths, but the latest log's side-table value is rounded to 0.000 GiB despite a note naming large-looking tensors. The exact accounting status of those tensors is not proven by the log.
- `tensor_bytes()` walks one hop through an object and merges overlapping ranges. It deliberately does not traverse arbitrary object graphs, so it can under-report allocations not reachable in one hop and can miss driver-managed buffers.
- The planner's fixed attention/Triton/fragmentation values can overlap with actual measured allocator/backend memory or can undercount it. Current diagnostics do not establish exact exclusivity.
- KV `kv_cost()` owns only each KV family's own buffers; sibling GDN state, page tables, graph pools, and workspaces are charged elsewhere. This division is valid only if every buffer has exactly one owner.

## 11. What is knowable before final cache commitment

### A. Exactly knowable before cache allocation

- GPU device total and driver free memory at a measurement point, subject to other-process races.
- Model configuration dimensions.
- Dtype item size.
- Expert source tensor shapes/dtypes when banks are fully loaded/materialized.
- KV bytes per page from backend geometry, including Turbo3/Turbo4 code/norm layout.
- Page token count and dummy-page policy.
- Requested total context.
- Configured maximum prefill chunk.
- GDN state-pool geometry and slot count.
- Page-table dimensions.

### B. Calculable from model/backend geometry

- Expert row bytes for each distinct bank geometry.
- KV page bytes.
- Required KV pages for a context.
- Formula-based GDN tensor sizes if runtime batch, dtype, padding, and lifetimes are fully known.
- Page table size.
- State pool size.
- Some graph buffer sizes if capture shapes and allocator behavior are deterministic.

### C. Measurable through a safe pre-final-cache probe

- Actual attention backend initialization workspace.
- Actual Triton/JIT/autotune peak for a controlled shape.
- Actual CUDA graph capture cost for a selected graph set.
- Actual cache auxiliary tensor bytes after materialization.
- Actual allocator reserved/allocated/free relationship after mandatory setup.

### D. Only measurable during a real or representative prefill

- Actual FLA/GDN kernel temporary peak and liveness.
- Actual packed-sequence `NT` and runtime tensor shapes.
- Kernel-specific driver allocations and allocator block behavior.
- Request-dependent activation peaks.

### E. Fundamentally variable at runtime

- Other processes using the GPU.
- Routing-dependent expert-copy buffers and miss behavior.
- Request batch shape and sequence packing.
- CUDA allocator fragmentation and block reuse.
- Backend/JIT behavior after cache or kernel compilation changes.
- Multimodal bursts if enabled.

## 12. Dry-run/probe and two-phase architecture feasibility

The existing architecture makes a two-phase design technically plausible but not currently implemented.

Supporting facts:

- Model weights can be loaded before runtime pools.
- KV pool classes expose `kv_cost()` before pool creation.
- `OffloadMoeCache` has `rebuild()` and frees/recreates GPU bank caches, so cache resizing is conceptually supported while idle.
- KV pool families expose `rebuild_from_config()` and the scheduler has idle-only runtime rebuild support.
- CUDA graphs are created after caches and can be recaptured during runtime rebuild paths.
- The engine already has a transient probe base and peak measurement hooks.

Obstacles:

- The first real GDN prefill is currently needed to reveal the exact failing `h` allocation.
- The current calibration occurs after final cache sizing, so it cannot protect the first request.
- Some JIT/autotune allocations may be created only on the first invocation.
- CUDA Graph capture must be performed after final cache/model bindings and can itself alter memory pools.
- Rebuilding all pools may require clearing page/radix prefix state, rethreading managers, rebinding model scratch, and recapturing graphs.
- Some model auxiliary tensors are materialized as part of cache setup and are not currently represented before planning.
- A probe that uses the same large cache plan can itself OOM; a minimal-cache probe is necessary.

A technically plausible sequence is:

1. Load weights and mandatory model/backend structures.
2. Create minimal or zero/low-slot expert and KV pools that permit a probe.
3. Initialize attention backend.
4. Run controlled Triton/autotune and representative GDN prefill probes at candidate chunk sizes.
5. Record driver free, PyTorch allocated/reserved, peak allocated/reserved, and any backend diagnostics.
6. Destroy/rebuild probe pools and transient objects.
7. Compute the final negotiable pool from the physical post-probe baseline.
8. Create final expert/KV pools and, if enabled, graph capture.
9. Validate final post-capture headroom before readiness.

This is architecture-level feasibility only. The current code does not prove that every probe allocation can be completely reclaimed without reconstructing the process.

## 13. Cache reversibility

### `OffloadMoeCache`

`python/freetoken/moe/offload_cache.py` supports `rebuild(cache_size)` and clears/reallocates GPU cache tensors. It also has reset operations for routing/bookkeeping. Rebuild is intended for idle operation and is not proof that every auxiliary/JIT allocation is reclaimed.

### KV pools

`BaseKVCache` defines `rebuild_from_config()`. Pool implementations use `torch.cuda.empty_cache()` around replacement in several families. The scheduler's runtime rebuild requires no pending prefill and no running decode, and rebuilds page/token managers when needed.

### CUDA Graph pools

Graph capture creates private CUDA graph memory pools. The graph runner is recaptured during some runtime rebuild paths, but the current source does not establish complete destruction/reclamation of every old graph pool before a new final plan.

### Attention backend

The attention backend has setup and capture hooks. Its workspace ownership/release semantics vary by backend; the current planner uses a fixed workspace estimate rather than a universal measured interface.

### Auxiliary expert tensors

Auxiliary tensors are attached to cache/bank structures. They may be recreated with cache setup, but the current source does not establish a single owner and complete release protocol for every quantization-specific auxiliary allocation.

Conclusion: pool resizing is partially supported, but a safe calibration-before-final-sizing flow would need explicit ownership and release guarantees for backend, graph, JIT, and auxiliary allocations.

## 14. Relevant tests

### `tests/engine/test_vram_ledger.py`

Tests pure arithmetic for:

- ceilings and reserve behavior;
- context demand;
- `plan_for_context()` priority inversion;
- unaffordable context rejection;
- expert/KV budget constraints.

These tests do not run a real CUDA forward, do not measure driver free VRAM, and do not test allocator fragmentation or JIT/autotune coexistence.

### `tests/engine/test_cache_budget.py`

Tests pure integer byte arithmetic for:

- dummy pages;
- expert slot floors;
- page calculations;
- ratio compatibility;
- budget rejection.

They do not test actual tensor allocation or physical free memory.

### `tests/kvcache/test_pool_sizing_surface.py`

Tests pool sizing interfaces and configuration behavior. It does not certify a real 16K forward.

### `tests/moe/test_offload.py`

Tests CPU/fake-device/lazy/offload cache behavior, cache sizes, geometry, and rebuild behavior. It does not prove that a full Flash Next NVFP4 GPU cache fits together with the real GDN first prefill.

### `tests/kvcache/test_turbo_pool.py`, `tests/kvcache/test_turbo_kv.py`, `tests/kvcache/test_turbo_attn.py`, `tests/kernels/test_turbo_attention.py`

Test Turbo3/Turbo4 packing, decoding, dimensions, and numerical behavior. They do not test full-engine VRAM competition with experts, GDN, autotune, and a 16K request.

### MTP suites

The canonical scheduler/model suites exercise MTP arithmetic, rollback, and GDN equivalence. The reported real MTP equivalence result used 4K and replay behavior; it is not a 16K automatic planner test.

### CI

The latest recorded `make ci` passed with approximately 1,957 tests passed, 206 skipped, and 15 deselected. This proves source/test regressions were not detected by the normal suite. It does not prove 16K real GPU support.

### Missing test coverage

There is no current automated test proving:

- final physical free VRAM after automatic planning;
- `memory_allocated` plus `memory_reserved` plus non-PyTorch reconciliation;
- real 16K Turbo3 prefill;
- real 16K Turbo4 prefill;
- GDN chunk solver behavior;
- first-request Triton/autotune peak;
- allocator fragmentation under final cache sizing;
- persistent calibration reuse/invalidation;
- repeated startup/rebuild with complete reclamation;
- MTP k=1..6 at final 16K automatic settings.

## 15. Contradictions and unavailable evidence

1. Earlier descriptions identified `torch.zeros_like(v)` in `chunk_o.py` as the failed 192 MiB allocation. The newest log identifies `h = k.new_empty(...)` in `chunk_delta_h.py:293`. The newest stack is authoritative.
2. The ledger reports `measured:allocator-held = 13.659 GiB`, while the OOM later reports 14.73 GiB PyTorch allocated and 15.20 GiB process memory. These are different times and measurement domains; the current logs do not provide an instantaneous continuous trace between them.
3. `cache:expert-side-tables` prints 0.000 GiB but its note names tensors with apparently large individual sizes. The total byte value and ownership are not printed, so the note cannot be used to assert a 1.926 GiB side-table charge.
4. The latest successful server readiness proves initialization and planning completed, not that the context generated successfully.
5. Later attempted benchmark runs did not create valid JSON results; they cannot be used as success or failure proof for the final planner.
6. The exact runtime shape/dtype of `h` is absent from the OOM log.
7. `memory_reserved` and `max_memory_reserved` are not recorded in the startup/calibration log, so the allocator's full reserved state at the failure is unavailable.
8. No persistent calibration profile keyed by GPU/model/KV/backend exists in the inspected source.

## Evidence Package for ChatGPT

1. **Root failure mechanism supported by evidence.** The planner commits persistent expert/KV pools against a budget whose reserve model is not a proven upper bound for the first real GDN prefill. The first prefill then requests a 192 MiB GDN `h` allocation when only 171.38 MiB driver-free VRAM remains.

2. **Exact point where the planner commits too much VRAM.** `python/freetoken/engine/engine.py:1012-1024` calls `_resolve_auto_moe_cache_size()` before `OffloadMoeCache` construction and before the first real prefill. `python/freetoken/engine.py:1025-1047` then allocates the final expert cache and attaches auxiliary structures. `python/freetoken/engine.py:454-461` creates the KV pool from the selected page count. The plan leaves effectively zero ledger headroom: 1,262 expert slots, 259 KV pages, and a reported uncommitted value near zero.

3. **Exact memory missing or incorrectly estimated, if known.** The exact missing amount is not fully known. The immediately demonstrated deficit is at least 20.62 MiB because the request was 192.00 MiB and driver-free memory was 171.38 MiB. The unmodelled/incorrectly modelled portion includes the real FLA `h` allocation/lifetime and potentially other simultaneously live GDN/backend/allocator/non-PyTorch memory. The exact tensor shape and dtype are unavailable.

4. **Whether the 192 MiB allocation has been identified.** Yes, by source location: `python/freetoken/kernel/fla/chunk_delta_h.py:293`, `h = k.new_empty(B, NT, H, V, K)`. Its exact runtime shape and dtype have not been logged.

5. **Predicted versus real reconciliation.** Predicted: 15.179 GiB committed display, 13.922 GiB ceiling, 1.257 GiB modelled reserve, 0.000 GiB headroom, and 13.659 GiB allocator-held at calibration. Real OOM: 14.73 GiB PyTorch allocated, 84.28 MiB PyTorch reserved-but-unallocated, 15.20 GiB process use including non-PyTorch memory, 171.38 MiB driver free, and a failed 192 MiB allocation. The domains and timestamps do not reconcile exactly from existing evidence.

6. **Memory consumers calculable exactly.** Model geometry; dtype sizes; fully materialized expert row geometry; Turbo3/Turbo4 KV bytes per page; page count plus dummy page; GDN state pool geometry; page-table dimensions; configured scheduler chunk; requested context. Exactness still depends on actual runtime shapes and allocation ownership.

7. **Memory consumers requiring empirical measurement.** Triton/JIT/autotune allocations; attention backend workspaces; CUDA graph private pools; PyTorch reserved blocks; driver/non-PyTorch allocations; allocator fragmentation; exact FLA temporary liveness; auxiliary quantization buffers whose ownership is not exposed; other-process memory.

8. **Whether prefill chunk can safely become a solved variable.** Architecturally yes. The scheduler already separates total `max_seq_len` from one-forward `max_extend_tokens`, and `PrefillManager` chunks requests. A solver could preserve 16K total KV capacity while selecting a smaller chunk, provided all backend minimum/alignment constraints and state continuity are respected. The current planner does not solve this variable.

9. **Whether a two-stage calibration/planning architecture is feasible.** Likely feasible but unproven. The engine can load weights, initialize minimal structures, run controlled probes, measure peaks, and rebuild idle caches. Obstacles are complete reclamation of graph/JIT/backend pools, auxiliary ownership, and proving that a minimal probe exercises the same peak path as the final configuration.

10. **Source files/functions requiring modification for a complete fix.** Likely areas are:

   - `python/freetoken/engine/engine.py`: initialization ordering, `_open_vram_ledger`, `_resolve_auto_moe_cache_size`, `_calibrate_vram_ledger`, final pool creation, and probe/rebuild orchestration;
   - `python/freetoken/engine/vram_ledger.py`: charge ownership, driver/allocator domains, transient peak model, and removal of heuristic-only policy;
   - `python/freetoken/engine/cache_budget.py`: exact solver over context, chunk, expert slots, KV pages, and fixed/temporary consumers;
   - `python/freetoken/scheduler/config.py`, `scheduler.py`, `prefill.py`: make prefill chunk a solver output while preserving total context;
   - `python/freetoken/models/qwen3_5_moe/gdn_kernels.py`, `python/freetoken/kernel/fla/chunk_delta_h.py`, `chunk.py`, and related GDN code: expose exact runtime shapes/temporary requirements;
   - `python/freetoken/engine/graph.py`: measured graph pool ownership and lifecycle;
   - attention backend modules and `kvcache/*`: exact fixed/scratch cost ownership;
   - `python/freetoken/moe/offload_cache.py`: full auxiliary/bookkeeping ownership, reversible allocation, and pre-plan byte reporting;
   - calibration/profile storage code, if persistent calibration is retained.

11. **Remaining unknowns requiring targeted GPU experiment.** Exact `h` shape/dtype/`NT`; peak allocated and reserved around the first GDN forward; driver free immediately before and after each major allocation; complete expert auxiliary tensor total; attention and Triton allocations with graphs disabled; allocator reserved blocks; whether a smaller chunk avoids OOM while preserving 16K; whether graph-enabled configurations introduce a second independent failure.

12. **Smallest experiments necessary to remove unknowns.**

   1. Run the same 16K Turbo4 configuration with a diagnostic-only wrapper/logging path that records `B, NT, H, V, K, dtype`, `memory_allocated`, `memory_reserved`, `max_memory_allocated`, `max_memory_reserved`, and `mem_get_info` immediately before GDN and after each layer; do not change the planner in that experiment.
   2. Run the same configuration with controlled `max_extend_tokens` values that are valid for the scheduler, recording whether total 16K KV capacity remains available and whether the first prefill completes.
   3. Repeat with Turbo3 to separate GDN/backend pressure from KV page cost.
   4. Repeat with minimal expert cache and then with the automatic expert cache to isolate the exact persistent pool contribution.
   5. Compare tensor-walk bytes, PyTorch allocated/reserved, and driver free after initialization and immediately before GDN.
   6. Run a graph-disabled probe first; only after that succeeds, repeat with the intended graph configuration.
   7. Test an idle cache rebuild after the probe and verify that old expert/KV/graph allocations are physically reclaimed before final sizing.

No complete real 16K generation succeeded in the inspected evidence. The report therefore does not claim 16K support.
