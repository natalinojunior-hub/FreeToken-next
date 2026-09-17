DFlash draft: 6 decoder layers + `fc.weight` (projects target hidden features → n_embd; dflash_n_target_features=25600 default, target_layer_ids 8 layers) + `enc.output_norm.weight` (encoder norm over target features) + `output_norm.weight`. No token_embd/output → borrows from the target.

Now I'm ready. Let me write the dense markdown report (~200 lines).

Let me double-check a few file:LINE cites I plan to use:

Part A:
- /models/desenvolvimento/freetoken-next/python/freetoken/models/qwen4_exp/weight.py:89 `if raw_name.startswith("mtp."): return None`
- weight.py:9 docstring "Dropped: mtp.* (speculative head...)"
- weight.py:39-40 anchor excludes MTP experts
- qwen3_5_moe/weight.py:58
- glm5_next/weight.py:48, glm_moe_dsa/weight.py:7, glm4_moe/weight.py:122, config.py:28, deepseek_v4/weight.py:178
- deepseek_v4/args.py:33 n_mtp_layers (dead)
- kvcache/qsa_pool.py:49-51
- kernel/fla/fused_sigmoid_gating_recurrent.py:29,56,263-280,319,367
- engine/engine.py:987-1005 (forward_batch: complete_one, single sample)
- engine/graph.py:79-99 fixed bs candidates
- server/args.py — no spec flags; only --ple-backend:619
- tests/models/qwen4_exp/test_weight.py:234-236, test_weight_ckpt.py:213-216
- tests/models/test_qwen3_5_moe_weight.py:72,171,342
- tests/models/qwen4_exp/common.py:272-273, 294, 332
- models/register.py:181-192
- models/qwen4_exp/config.py — Qwen4ExpArgs fields; no mtp
- model.py Qwen4ExpModel / lm_head
- ple.py 730 lines, ple_disk.py 283 lines; engine/config.py:33-34 ple_backend="disk"; engine.py:1284 pin budget; model.py:load_host_tables asserts len(ple_layers)==1 for disk.

Git branches: upstream/feat/ple-disk, upstream/fix/ftw-ple-table, upstream/feat/decode-token-checkpoint (interesting: "decode-token-checkpoint" might be spec-related? It's about checkpointing decode tokens — the `--enable-special-token-ckpt`. Not spec decode.)

Note current branch is `next` == main == v0.1.3, and upstream/feat/ple-disk is remote-only. Is ple_disk.py already merged? It exists on disk in `next`, so yes merged.

Part B cites — I have them.

Let me also note llama-arch.cpp:221-222 KV keys, 583-591 tensor names, 982-993 classification.

And LLM_KV_PLE_* keys — I saw usage but not line numbers. Not critical; I'll reference qwen4exp.cpp:71-90 region for PLE keys.

Let me quickly get line numbers for LLM_KV_PLE_LAYERS definitions to be safe... I'll cite by file:qwen4exp.cpp:72-99 which I read (the get_key block starting at "// PLE n-gram hash embeddings" around line 70). Actually the read showed offsets 161-380 and 41-160. The PLE metadata read block is in 41-160 chunk (shown as lines ~70-140). I'll cite approximate ranges: qwen4exp.cpp:71-133. Fine, I'll say qwen4exp.cpp:72 (ml.get_arr_n(LLM_KV_PLE_LAYERS...)) — from the offset math: the chunk started at line 41; counting: line 41 = "qwen4exp_require_nonzero(ml, LLM_KV_SSM_TIME_STEP_RANK..." — the PLE block "uint32_t n_ple = 0;" appeared ~30 lines later → line ~73. Good enough; I'll cite ranges.

Let me write it now. Aim ~200 lines.
# MTP / speculative-decoding investigation

## A — FreeToken-next (`/models/desenvolvimento/freetoken-next`, pkg `python/freetoken/`)

### A1. Search results — the complete, honest list

Repo is on branch `next` @ `cac247a` (== `main` == `v0.1.3`). **Zero documentation** mentions speculative decoding / MTP (grep over `*.md|*.toml|*.yaml|*.json` for `speculat|MTP` → no files).

**Real functionality: NONE.** Every hit is one of three things:

| category | evidence |
|---|---|
| **MTP tensors are explicitly *dropped*** | `python/freetoken/models/qwen4_exp/weight.py:9` (docstring: *"Dropped: `mtp.*` (speculative head, including its stacked `mtp.layers.0.mlp.experts.*`)"*), `weight.py:39-46` (`_EXPERT_KEY_RE` anchored on `model.language_model.` to exclude MTP experts), `weight.py:89` (`if raw_name.startswith("mtp."): return None`) — same pattern in `models/qwen3_5_moe/weight.py:58`, `models/glm5_next/weight.py:11,48`, `models/glm_moe_dsa/weight.py:7`, `models/glm4_moe/weight.py:122`, `models/glm4_moe/config.py:28`, `models/deepseek_v4/weight.py:178`, `models/config.py:32` |
| **vestigial dead code / dead config** | `models/deepseek_v4/args.py:33` `n_mtp_layers: int = 1` — a dataclass field with **no reader anywhere** (`grep n_mtp_layers` → only that line). `kvcache/qsa_pool.py:49-51` `ring_capacity_for(index_ratio, num_speculative_tokens=0)` — always called with the default (`qsa_pool.py:76`, `:176`, `tests/kvcache/test_qsa_pool.py:118` is the only nonzero call, a unit test). `kernel/fla/fused_sigmoid_gating_recurrent.py:29,56,263-280,319,367` — vendored SGLang kernel still carries `target_verify` args (`intermediate_states_buffer`, `retrieve_parent_token`, `HAS_EAGLE_TREE_CUSTOM_ATTN_MASK`) and a stale `--speculative-adaptive` comment; **no caller in `python/` passes any of them** (grep for those names outside `kernel/fla/` → 0 hits). |
| **unrelated word usage** | `models/muse_glimmer/model.py:111` ("the DFlash drafter embeds without the norm" — a *reason* to keep `embed_norm` unfused, not a drafter); `engine/engine.py:935`, `scheduler/scheduler.py:642,672,682-692`, `kvcache/base.py:13-117` (`CacheRebuildRejected`) — "rollback"/"reject" are **runtime KV-rebuild** terms; `server/openai_api.py:45,167-172`, `server/accounting.py:48` — "accepted" = HTTP/arg validation; `engine/sample.py:24-80` `Sampler`/`sample_impl`, `engine/graph.py:230` `_default_jobs(num_specs)` (= "specializations"). No `medusa`, `lookahead`, `eagle` head, `next_n`, `num_spec`, `accept_len` anywhere. |

`verify` hits: all prose except the FLA kernel above. `draft` hits: only the two lines above.

### A2. Loop map — **NOT PRESENT**

There is no draft/verify loop. The decode step is strictly one token per request per forward:

- `engine/engine.py:987-1005` `forward_batch()` → `graph_runner.replay(batch)` → `for req in batch.reqs: req.complete_one()` → `next_tokens_gpu = self.sampler.sample(batch_logits, args)`. `batch_logits = logits[:batch.size]` — exactly **one output row per request**, one sampled token, no accepted-length computation, no bonus token, no KV rewind of rejected tokens.
- CUDA graphs (`engine/graph.py:79-99` `_determine_cuda_graph_bs`) capture a fixed **batch-size** ladder `[1,2,4,8,16,…]` (`can_use_cuda_graph` `graph.py:204`, `pad_batch` `graph.py:215`). There is **no verify-length dimension** in the graph key, so variable-length verification is not representable today.
- CLI/server flags: `grep '"--' server/args.py | grep -i 'spec|draft|mtp|next|eagle|accept'` → **only `--ple-backend` (`server/args.py:619`) and `--enable-special-token-ckpt` (`:740`)**. `NOT PRESENT`: `--spec-type`, `--spec-draft-model`, `--num-speculative-tokens`, `--draft-*`.

### A3. qwen4exp ("Qwen3.8 Flash Next") architecture support — present, MTP-blind

Package `python/freetoken/models/qwen4_exp/` (2852 LOC): `config.py` (264), `model.py` (230), `attention.py` (250), `gdn.py` (203) + `gdn_reference.py`, `hc.py` (154), `ple.py` (730), `ple_disk.py` (283), `moe.py` (29), `weight.py` (394). Registered at `models/register.py:181-192` (`"Qwen4ExpForConditionalGeneration"`, comment: *36 GDN + 12 QSA compressed-sparse layers on 4 hyper-connection streams, a PLE n-gram embedding layer, 512 NVFP4 routed experts top-10 + gated shared expert*).

Config (`models/qwen4_exp/config.py`):
- `Qwen4ExpArgs` `config.py:22-67`: `hc_count`, `hc_lowrank`, `ple_layer_ids`, `ple_embed_dim`, `ple_conv_kernel_size`, `ngram_size`, `heads_per_ngram`, `ngram_vocab_size_base`, `make_ngram_vocab_size_divisible_by`, `split_ngram_parts`, `ngram_boundary_token_id` (= eos), `index_n_heads/kv_heads/head_dim/budget/ratio`.
- Two attention groups (`config.py:171-197`): `FullAttentionGroupConfig` (QSA, `index_head_dim=text.indexer_head_dim`, `index_ratio=text.indexer_compress_ratio`) + `LinearGatedDeltaGroupConfig` (GDN). Layer typing from `text.layer_types` or `full_attention_interval` (`_layer_types`, `config.py:95-107`), with `qwen_sparse_attention → full_attention` rewrite.
- PLE must sit on a `linear_attention` layer (`config.py:159-162`); per-request PLE state is declared via `ple_slot_states()` (`config.py:76-96`) as two `SlotStateSpec`s — `PLE_CONV_STATE` `[hc*hidden, (k-1)*ngram]` and `PLE_NGRAM_STATE` `[ngram_size-1]` int32 filled with eos.
- **No MTP field is read.** `grep -n "mtp\|num_nextn\|nextn\|speculat" models/qwen4_exp/config.py` → 0. `parse_config` ignores `mtp_num_hidden_layers`, `mtp_use_dedicated_embeddings`, and the nested `text_config.mtp` dict entirely.
- Model has a single `ParallelLMHead` (`model.py:126-134`), tied to `model.embed_tokens` when `tie_word_embeddings` (`config.py:236`), and one `hyper_connection_mixer` collapse before it (`model.py:108`, `model.py:150-168` `forward`).
- Tests **enforce** the drop: `tests/models/qwen4_exp/test_weight.py:234-236` (`assert not name.startswith("mtp.")`), `tests/models/qwen4_exp/test_weight_ckpt.py:213-216`, `tests/models/qwen4_exp/common.py:272-273,294,332` (quant groups that exist only for `mtp.layers.0.mlp.experts`), `tests/models/test_qwen3_5_moe_weight.py:72,171,342`.

### A4. PLE caching / disk offload — present and merged

`upstream/feat/ple-disk` and `upstream/fix/ftw-ple-table` exist as remote branches; the code is already on `next`.

- Backend selector `engine/config.py:33-34`: `ple_backend: str = "disk"` — *"disk reads rows from the checkpoint files per fill, pinned preloads the table into page-locked host RAM"*. CLI `--ple-backend` (`server/args.py:619`).
- Seam: `models/qwen4_exp/model.py:136-186` `load_host_tables()`. Disk path (`model.py:150-176`) builds `DiskRowTable(resolve_row_source(folder), constants, max_graph_rows=max(256, cuda_graph_max_bs), max_extend_tokens=...)`, asserts **exactly one PLE layer** because "one WAIT node per captured graph: the flag protocol supports a single consume", and installs `disk_table.forward_host_ctx` — the engine wraps every dispatch in it (`engine/engine.py:991` `with self.ctx.forward_batch(batch), self.model.forward_host_ctx(batch, use_graph)`).
- `ple_disk.py:1-4` docstring: a C++ store hashes n-gram windows and batch-reads rows from fp8 shard tensors into pinned staging; the *captured* lookup is a fixed-shape H2D copy + dequant. Hash windows are pure functions of `req.input_ids + device_len`, so radix/prefix hits, restores and COW forks need no bookkeeping. Env `FREETOKEN_PLE_IO_URING`, `FREETOKEN_PLE_SYNC=auto|wait|gate` (`ple_disk.py:30-31`).
- Memory interaction: the pinned path returns `table.bank.nbytes` and the engine subtracts it from the host-pin budget (`model.py:186`, `engine/engine.py:355`, `engine/engine.py:1284-1318` — "``reserved`` subtracts host bytes already pinned outside the expert banks (qwen4_exp's PLE table)", 40 % WSL cap, `FREETOKEN_PIN_BUDGET_GB`). The 47.7 GiB table is 128 checkpoint shards (`weight.py:56-60`, `load_ple_table`).
- Interaction with spec decode: none today — but note `forward_host_ctx` and the fixed-shape graph rows are keyed on `cuda_graph_max_bs`/`max_extend_tokens`, i.e. exactly the two quantities a verify batch would inflate.

---

## B — Reference: `/models/servers/llama-turbo-optimal` (branch `perf/lto-perf-v2` @ `d0a274167`)

### B5. qwen4exp MTP implementation

**Arch/hparams + GGUF keys** — `src/models/qwen4exp.cpp` (1694 LOC):
- `qwen4exp.cpp:56-61` reads `LLM_KV_NEXTN_PREDICT_LAYERS` (`"%s.nextn_predict_layers"`, `src/llama-arch.cpp:221`) and hard-fails above one draft block: *"qwen4exp supports at most one MTP draft block"*.
- `src/llama-arch.cpp:222` `LLM_KV_NEXTN_SHARED_TARGET_TENSORS = "%s.nextn_shared_target_tensors"`.
- PLE keys (`qwen4exp.cpp:72-133`): `ple.layers` (only one layer allowed), `ple.ngram_size`, `ple.heads_per_ngram`, `ple.conv_kernel`, `ple.eos_token_id`, `ple.image_token_id` (optional), `embedding_length_per_layer`, `ple.layer_multipliers` (len = ngram_size), `ple.head_offsets`/`ple.head_vocab_sizes` (len = `ple_n_heads`, uint64 narrowed to int32 row index). HC keys `qwen4exp.cpp:45-54`: `hyper_connection.count`, `hyper_connection.low_rank`, with `n_embd_out_impl = hc_mult * n_embd` — **the MTP handoff width is `hc*n_embd`, not `n_embd`.**
- Layer typing `qwen4exp.cpp:141-151`: `attention.recurrent_layers` array, else every `full_attention_interval`-th layer.

**Tensor names → role** (`src/llama-arch.cpp:583-591`, all `blk.%d.nextn.*`, all classified `LAYER_REPEATING` per the comment at `llama-arch.cpp:982-984`):

| GGUF name | role | declared shape (`qwen4exp.cpp:328-342`) |
|---|---|---|
| `blk.N.nextn.enorm` | RMS gamma on the **token embedding**, width `n_embd` | `{n_embd}` |
| `blk.N.nextn.hnorm` | grouped RMS gamma on the **hc-wide residual**, width `hc*n_embd` | `{hc_dim}` |
| `blk.N.nextn.eh_proj` | concat(e‖h) → `n_embd`, the *only* fusion | `{2*n_embd, n_embd}` |
| `blk.N.nextn.hc_head_norm/down/up` | the draft's **own** output mixer (target's `output_hc_*` is not reused) | `{hc_dim}`/`{hc_dim,hc_lr}`/`{hc_lr,hc_dim}` |
| `blk.N.nextn.embed_tokens` | optional dedicated embedding | `{n_embd, n_vocab}`, `TENSOR_NOT_REQUIRED` |
| `blk.N.nextn.shared_head_head` | optional dedicated LM head | `{n_embd, n_vocab}`, `TENSOR_NOT_REQUIRED` |
| `blk.N.nextn.shared_head_norm` | declared for other archs (`llama-arch.cpp:588`), **not loaded by qwen4exp** | — |
| trunk | `token_embd.weight` / `output.weight` (with `output` falling back to a duplicated `token_embd`, `qwen4exp.cpp:181-185`) | |

**Sharing rule (the important one)** — `qwen4exp.cpp:495-499`:
```cpp
ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
```
and `qwen4exp.cpp:416`: `tok_embd_w = layer.nextn.embed_tokens ? … : model.tok_embd;` with the comment at `:335-336` *"Current Qwen4exp shares the trunk token embedding and LM head."* Sidecar borrowing is implemented in `src/llama-model-loader.cpp:1521-1570` (`borrow_shared_tensor`) — it only fires for `LLM_TENSOR_TOKEN_EMBD`/`LLM_TENSOR_OUTPUT`, only when `nextn_shared_target_tensors` is true, requires the tensor to be *absent* from the sidecar, and validates the target's shape dim-by-dim before aliasing (`throw` on disagreement, `:1541-1548`).

**Load modes** — `qwen4exp.cpp:171-175`: `mtp_only = ml.load_mtp && n_layer_nextn>0 && ml.get_weight("blk.0.hc_attn_norm.weight")==nullptr` → trunk tensors become `TENSOR_NOT_REQUIRED`. Draft block loads at `qwen4exp.cpp:286-342` with `flags = ml.load_mtp ? 0 : TENSOR_SKIP`; its indexer tensors are `flags|NOT_REQUIRED|SKIP|SKIP_IF_VIRTUAL` (`:311-318`) — *"The MTP block uses dense attention. Preserve optional indexer tensors … but never load or execute them"*. `:345-366` explicitly consumes every optional `scale`/`input_scale` name in SKIP mode so strict GGUF tensor accounting still passes.

**Draft graph** — `qwen4exp.cpp:376-378` dispatch on `params.gtype == LLM_GRAPH_TYPE_DECODER_MTP`; graph at `:386-505`. Sequence: `get_rows(tok_embd_w, tokens)` → grouped-RMS `h` (`reshape_3d(h, n_embd, hc, T)` → rms_norm → `mul hnorm`, `:425-433`) → RMS `e` then `repeat` to `hc` streams (`:437-441`) → `concat(e_norm, h_norm, 0)` → `eh_proj` → per-block `build_hc_mix`/`build_layer_attn(dense)`/`build_hc_combine` → `build_hc_mix`/`build_layer_ffn`/`build_hc_combine` → `hc_head` mixer → `head_w` GEMM → `result_output`. It also re-emits its own wide residual: `res->t_h_nextn = flat` (unmasked) or `flat_out` (masked) (`:480-484`) — this is what makes **recursive** single-head drafting work.

**Target handoff** — `qwen4exp.cpp:676-681`: `if (cparams.embeddings_nextn) { cb(res_hc,"h_nextn"); res->t_h_nextn = res_hc; }` placed **before** the final `output_hc_*` mixer, i.e. the draft sees the *wide* pre-collapse residual. `qwen4exp.cpp:586` `keep_full_nextn = embeddings_nextn && !embeddings_nextn_masked` and `:648` `if (il == n_layer-1 && inp_out_ids && !keep_full_nextn)` — the target **only crops output rows when it is not feeding an unmasked handoff**; draft chaining (`masked=true`) keeps the normal crop.
Extraction: `src/llama-context.cpp:4380`, `:4472-4479` (async D2H of `n_tokens*n_embd` floats), masked indexing `get_embeddings_nextn_ith` `:1486-1507` (unmasked = dense raw row index; masked = `logits==0` compacted). Flags set once in the drafter ctor: `llama_set_embeddings_nextn(ctx_tgt,true,false)` / `(ctx_dft,true,true)` (`common/speculative.cpp:2694-2696`).

**Drafter driver** — one generic implementation for all native-MTP models: `common/speculative_impl_draft_mtp` at `common/speculative.cpp:2591-3180`, registered at `:5786`. Mode table in the comment at `:2600-2606`: `is_mem_shared` (gemma4) / `chain_heads` (step35, `n_mtp_layers>1`) / *neither* = **one recursive head**, which is qwen4exp. Width guard at `:2649-2651`: `n_embd = llama_model_n_embd_out(ctx_dft)` must equal the target's `n_embd_out()` — "MTP input row width must match the target h_nextn width".

Loop:
1. **Target prefill/verify replay → draft KV catch-up** (`process()`, `:2773-2937`). Rewind first: `llama_memory_seq_rm(mem_dft, seq_id, batch_in.pos[i_batch_beg], -1)` (`:2846-2848`) to the verified frontier. Then rebuild a token batch over the *same* positions with `embd` **shifted right by one** (`:2863-2867`: `memcpy(batch.embd + 1*n_embd, h_tgt, row_bytes*(n_tokens-1))`) and row 0 filled from `pending_h[seq]` (`:2869-2887`) — the pair is `(h_p, x_{p+1})`. Then one `llama_decode(ctx_dft, batch)` commits draft KV for all those positions.
   Recovery boundary: if the carry is stale (restored/rewound sequence), `common_speculative_mtp_process_preflight_resolve` returns `target_only` (`:2813-2838`) → **drop the whole draft seq** (`seq_rm(-1,-1)`), keep only the newest target row into `pending_h`, reset `verify_h_rows=0`, and re-arm.
   After decode, `verify_h[seq]` caches all target rows of this batch and `pending_h = verify_h[n_rows-1]` (`:2919-2933`).
2. **Draft** (`:2942-3109`). `llama_memory_seq_rm(mem_dft, seq_id, dp.n_past, -1)` truncates stale draft positions first (`:2966-2969`). Batch row = `(dp.id_last, dp.n_past, embd=carry)` (`:2982`). Then `while (n_drafting)`: `common_sampler_sample(smpl, ctx_dft, i_last[seq], true)` → greedy-ish top-k chain; **stop if `cur_p->data[0].p < params.p_min`** (`:3037-3043`); push token; stop at `n_max_eff`. Recursive (non-shared, non-chained) step advances the position: `common_batch_add(batch, id, dp.n_past + i + 1, …)` with `embd = h_row` from the draft's own `t_h_nextn` (`:3076-3078`). Finally drafts shorter than `n_min` are cleared (`:3100-3102`).
   Adaptive depth: `GGML_MTP_DRAFT_ADAPTIVE` (`:2706-2711`) — only when `n_mtp_layers==1 && !is_mem_shared && n_max==3`; probes depth-3, and after 16 attempts with marginal acceptance `<0.50` caps that sequence at 2 (`accept()`, `:3111-3131`).
3. **Verify** (target): built by the caller. `common_sampler_sample_and_accept_n` (`common/sampling.cpp:828-855`, impl `:796-825`) — `idxs.size()==draft.size()+1`; for `i<draft.size()` it samples at output row `i` and **breaks on the first mismatch**; if all matched it samples one extra (bonus) row. This is **greedy matching against the target's own sampler chain** — no rejection-sampling ratio test. `common_sampler_accept_draft` (`:836-845`) is the same comparator fed pre-sampled logits (backend-offload / argmax fast path).
4. **Accept / rollback.** `n_accepted = ids.size()-1` (`tools/server/server-context.cpp:19939`); `rollback_depth = n_draft - n_accepted_draft` (`:19949`). `common_speculative_accept(...)` (`:19974-19976`) → MTP `accept()` re-points the carry at `verify_h[min(n_accepted, n_rows-1)]` (`speculative.cpp:3139-3141`). Draft-KV truncation is `common_speculative_rollback_dft()` → `llama_memory_seq_rm(mem_dft, seq_id, n_past, -1)` with the explicit note (`:6735-6738`) that **calling `accept()` twice corrupts the carry contract**. Target KV rollback uses the draft-backup/tape path (`server-context.cpp:19985+`). Ordering rule is stated in code: `update_logits` must run **before** rollback and before `spec_draft` is cleared (`:19918-19931`).

**KV topology (critical for qwen4exp)** — `src/llama-model.cpp:2624-2629`: `mtp_on_hybrid_qwen = ctx_type==MTP && arch ∈ {qwen3next,qwen35,qwen35moe,qwen4exp,bailingmoe3}` → the MTP context uses a **plain `llama_kv_cache`, not `llama_memory_hybrid_idx`**, with `layer_filter = il >= hparams.n_layer()` (`:2632` and the GLM_DSA/DEEPSEEK32 twin at `:2450-2480`). Consequence: **the draft has no GDN recurrent state, no PLE conv/n-gram state, and no QSA indexer cache** — it is a single dense-attention + MoE block. The trunk context's filters exclude the nextn layer from attn/recr/indexer (`:2663-2677`). Context creation refuses MTP on a model without MTP: `src/llama-context.cpp:7394` `if (ctx_type==MTP && !llama_model_has_mtp(model))`. `has_mtp()` = `n_layer_nextn>0 && router_layer<0` (`llama-hparams.h:460-463`; Granite Switch deliberately excluded). `n_layer()` = `n_layer_all - n_layer_nextn` (`llama-hparams.cpp:323`). VBR is disarmed on the MTP ctx (`llama-context.cpp:361-378`): *"An MTP self-draft carries its own extra (nextn) KV layer but shares the target's backbone KV"*.

**Graph/CUDA handling of variable verify length.** This fork has no CUDA-graph backend in `llama/`; it uses **ggml graph reuse** (`src/llama-graph.h:959` `can_reuse` compares `cparams.embeddings_nextn`/`embeddings_nextn_masked`, so a nextn-hooked graph never aliases a plain one; `llm_graph_input_embd_h::can_reuse` `:208`). Variable length is absorbed by `n_outputs_max` / `n_outputs_max_per_seq` (`llama-context.cpp:321-323`, sizing at `:5182-5207`: `embd_nextn.size = n_embd_out * n_outputs_max`) and by node budget `graph_max_nodes` (`llama-context.cpp:5486` gives qwen4exp `n_tokens*40`). `mtp_h_input` is a named graph input (`qwen4exp.cpp:414`) so the h-row buffer address is stable across replays. Note `qwen4exp.cpp:389` `GGML_ASSERT(ubatch.token)` — the MTP graph **cannot run in embedding-input mode**.

**Flags** — `--spec-type {draft-mtp|mtp}` (`common/speculative.cpp:44,57`; `common/arg.cpp:5404`), `--spec-draft-model/-md` (`arg.cpp:5389`), `--spec-draft-n-max` (`:5266`), `-n-min` (`:5277`), `--spec-draft-p-min` (`:5335`), `--spec-draft-backend-sampling` (`:5343`), `--spec-draft-caches` (`:5216/5229`), `--spec-mtp-vocab-size {0|32768}` (`:5284-5293`, backed by `common/mtp-vocab-trim.cpp` — FR-Spec-style vocab trim of a supported standalone Qwen-27B MTP GGUF, cached derivative, never mutates the source; balanced vocab in `common/mtp-vocab-qwen27b-balanced.inc`). Auto-select of `draft-mtp` when the target declares nextn layers: `speculative.cpp:5321`. Native-vs-sidecar: `combined_external_and_mtp` creates `ctx_mtp` from the **target model** (`:5648-5657`), else a `ctx_dft` from the draft model with `cparams_mtp` (`:5596-5645`). Env `GGML_MTP_DRAFT_ADAPTIVE`. Graph-level API: `llama_set_embeddings_nextn(ctx, value, masked)`, `llama_get_embeddings_nextn[_ith]`, `llama_set_nextn_layer_offset` (`src/llama-ext.h:228-240`).

**Other drafters present** (for contrast): `src/models/dflash_draft.cpp`, `gemma4_dflash_draft.cpp`, `eagle3.cpp`; DFlash hparams at `llama-hparams.h:204-213` (`dflash_block_size`, `dflash_mask_token_id`, `dflash_n_target_features`, `dflash_target_layer_ids[8]`, conv/selector fields), driven by `common_speculative_impl_draft_dflash` (block-parallel masked-token drafting, batch built at `speculative.cpp:6587-6592`, argmax/top-K extraction `:6609-6646`).

### B6. Correctness checklist for a new implementation

1. **Read `nextn_predict_layers` and support exactly 1.** Reject >1 with a config error (B-`qwen4exp.cpp:56-61`). Effective trunk length = `n_layer_all - n_layer_nextn`; the draft block index is `n_layer` (i.e. one past the trunk, `qwen4exp.cpp:393` `const int il = hparams.n_layer()`).
2. **Handoff width is `hc_count * hidden_size`, not `hidden_size`.** Assert target `n_embd_out == draft n_embd_out` before any drafting (`speculative.cpp:2649-2651`). The row taken from the target is the **wide residual before the final output mixer** (`qwen4exp.cpp:676-681`), never post-`output_hc` / post-`model.norm`.
3. **Tensor map is exact and ordered:** `enorm`(`n_embd`) applies to `embed_tokens(x)`; `hnorm`(`hc*n_embd`) applies to the grouped-RMS-normalised wide residual (**per-stream reduction, one shared concatenated gamma** — `qwen4exp.cpp:423-434`); `eh_proj` is `mul_mat([2*n_embd, n_embd], concat(e‖h, dim0))` with `e` **repeated across the hc streams** first (`:437-441`). Getting the concat axis or the repeat wrong is silent, not fatal.
4. **Head/embedding sharing must be resolved by fallback, not assumption:** `nextn.embed_tokens → model.tok_embd`, `nextn.shared_head_head → model.output` (`:416`, `:497-499`). For `qwen4exp` in practice the dedicated tensors are absent (`TENSOR_NOT_REQUIRED`), and `nextn.shared_head_norm` is **not** loaded — do not apply it. A sidecar with `nextn_shared_target_tensors=true` must alias the loaded target tensor and validate the shape, and must error if the target cannot provide it (`llama-model-loader.cpp:1521-1570`).
5. **The draft block is dense attention.** Never build/run the QSA indexer path in the draft graph (`qwen4exp.cpp:311-318`); it must not read the trunk's indexer cache.
6. **No recurrent/PLE state in the draft.** The draft's memory is a plain attention KV cache filtered to `il >= n_layer` (`llama-model.cpp:2624-2632`). The GDN conv/state pool and PLE conv+n-gram windows are *not* advanced by drafting and must *not* be rolled back by it.
7. **Carry pair semantics:** the draft consumes `(h_p, x_{p+1})`. When replaying a target batch, shift the target h-rows **right by one** and source row 0 from the cross-batch `pending_h` (`speculative.cpp:2863-2887`). After a full verify, `pending_h = verify_h[n_rows-1]`.
8. **Commit-or-discard, not both.** Before replaying target rows, `seq_rm(draft, seq, first_pos, -1)`; before each `draft()`, `seq_rm(draft, seq, n_past, -1)`; after acceptance, `common_speculative_rollback_dft` → `seq_rm(draft, seq, n_past, -1)`. Never `seq_rm` the *target* from the drafter. Positions must be non-decreasing vs stored (M-RoPE) — `speculative.cpp:2841-2843`.
9. **Accept = greedy prefix match on the target's own sampler output**, at output rows `0..n_draft` of the verify batch (`sampling.cpp:796-833`). The target row index list `idxs` must be `draft.size()+1` long and must come from the *same* batch that produced the draft tokens. Only the first mismatch position contributes a correction token; everything after is discarded.
10. **Target sampler RNG/temperature is untouched by verify.** The target chain is sampled once per output row and `common_sampler_accept` is called for each, so the chain (incl. grammar and any penalties) advances exactly as in non-speculative decoding. Speculation must not re-derive the distribution, must not resample, and must not use its own temperature for verification. Draft-side sampling is a separate chain: `top_k=10` (`speculative.cpp:2672-2677`) or a backend chain with `top_k(10)` (`:2681-2692`), and only the **argmax/candidate-0** token is drafted, gated by `p_min`.
11. **Carry `pending_h` is process-local, not sequence state** — `get_state`/`set_state` serialise it separately (`:3144-3159`), and `sequence_transition()` resets it + `verify_h_rows` + `i_last` + adaptive counters (`:3160-3175`). A restored/rewound frontier must **not** replay a target batch — take the `target_only` recovery path instead (`:2808-2838`).
12. **Never double-accept.** `accept()` updates the carry exactly once per verify (`speculative.cpp:6735-6738`).
13. **EOS / truncation:** a zero-accepted verify is still a controller outcome and must call `accept(0)` (`server-context.cpp:19971-19976`); accepted tokens are appended as `ids[0..n_accepted]` with the bonus token included (`:19978-19980`) and the transient target-cache truncation is `prompt.n_tokens() - n_draft` (`:19979-19981`). A draft shorter than `n_min` is discarded before verify (`speculative.cpp:3100-3102`); `p_min` can reject the *first* candidate, which must leave drafting disabled for that step without corrupting KV (`:296` comment).
14. **Graph identity:** any captured/reused graph that exposes `h_nextn` is keyed on `embeddings_nextn` + `embeddings_nextn_masked` (`llama-graph.h:959`); target runs unmasked, the draft runs masked (`:2694-2695`). `t_h_nextn` must be marked an output (`llama-graph.cpp:1675-1676`) and the input h-row must be a stable address (`ggml_set_name(inp->h,"mtp_h_input")`).
15. **Optional-tensor accounting is strict:** declare/consume every `*.scale` / `*.input_scale` sibling in SKIP mode so `--check-tensors` still passes (`qwen4exp.cpp:345-366`).

---

## C — Files on disk

Tooling used: `/models/servers/freetoken/venv/bin/python` + `gguf` 0.19.0 (`gguf.gguf_reader.GGUFReader`, `gguf.constants.GGMLQuantizationType`), names aggregated by regex family; metadata-only via `gguf-dump --no-tensors` (note: `-t` is not a valid flag in this version). No tensor payload was read.

### C.1 `/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-MTP-Q4XS-Q3S.gguf`
11.15 GiB · 866 tensors · **arch `qwen35`** (dense/hybrid GDN, *not* qwen4exp) · `general.file_type=26` (IQ3_S mix).
Geometry: `block_count=65`, `head_count=24`, `head_count_kv=4`, `embedding_length=5120`, `feed_forward_length=17408`, `context_length=262144`, `vocab=248320`, `rope.dimension_count=64`, `rope.dimension_sections=[11,11,10,0]`, `rope.freq_base=1e7`. 48 GDN layers (`ssm_*`, `attn_qkv`, `attn_gate`) + 17 full-attn layers (`attn_q/k/v/o`, `attn_q_norm`/`k_norm` `256`).
`qwen35.nextn_predict_layers = 1` → **MTP tensors (4, at `blk.65`):** `nextn.eh_proj.weight [10240,5120] IQ4_XS`, `nextn.enorm.weight [5120] F32`, `nextn.hnorm.weight [5120] F32`, `nextn.shared_head_norm.weight [5120] F32`. **No `nextn.embed_tokens`, no `nextn.shared_head_head`** → shares `token_embd.weight [5120,248320] IQ2_S` and `output.weight [5120,248320] Q4_K`. Note `hnorm` width = `n_embd` (no hyper-connections in this arch) and the file carries `shared_head_norm`, which qwen35 (not qwen4exp) does consume.

### C.2 `/models/Ornith-1.5-35B-A3B-APEX-MTP-I-Compact.gguf`
16.24 GiB · 753 tensors · **arch `qwen35moe`** · `file_type=15` (Q4_K_M).
`block_count=41`, `head_count=16`, `head_count_kv=2`, `embedding_length=2048`, `expert_count=256`, `expert_feed_forward_length=512`, `expert_shared_feed_forward_length=512`, `context_length=262144`, `vocab=248320`. 30 GDN + 11 full-attn layers.
`qwen35moe.nextn_predict_layers=1` → **MTP (4, `blk.41`):** `nextn.eh_proj.weight [4096,2048] Q8_0`, `enorm [2048] F32`, `hnorm [2048] F32`, `shared_head_norm [2048] F32`. Shares `token_embd.weight [2048,248320] Q4_K` / `output.weight [2048,248320] Q6_K`. MTP head is **higher precision (Q8_0) than the trunk (Q4_K_M)** — the "APEX … MTP-I" naming.

### C.3 `/models/Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf`
17.35 GiB · 753 tensors · **arch `qwen35moe`**, `general.name = Ornith-1.5-35B` (same base as C.2) · `file_type=30`.
Geometry identical to C.2 (41 layers, 16/2 heads, 2048, 256 experts).
`qwen35moe.nextn_predict_layers=1` → **MTP (4, `blk.41`):** `nextn.eh_proj.weight [4096,2048] Q8_0`, `enorm/hnorm/shared_head_norm [2048] F32`. Shares `token_embd [2048,248320] Q8_0` / `output.weight [2048,248320] Q6_K`. All 11 `attn_output` tensors are Q8_0 (uniform-demand "UD").

### C.4 DFlash drafts
`/models/qwen36-35b-a3b-dflash-Q4_K_M.gguf` — 0.22 GiB, 69 tensors, **arch `dflash`**, name "Qwen3.6 35B A3B DFlash", `file_type=15`.
`/models/dflash-draft-Ornith15.gguf` — 0.73 GiB, 69 tensors, **arch `dflash`**, name "Dflash_Draft", `file_type=32`.
Both: `block_count=6`, `embedding_length=2048`, `head_count=32`, `head_count_kv=8`, `key_length=value_length=128`, `feed_forward_length=6144`, `context_length=262144`, `attention.sliding_window=4096` with `sliding_window_pattern=[T,T,T,T,T,F]`, **`dflash.block_size=16`**, **`dflash.target_layers=[2,7,12,17,23,28,33,38]`**, `tokenizer.ggml.mask_token_id=248077`.
Tensor families (identical in both): `blk.{0..5}.{attn_q,attn_k,attn_v,attn_output,attn_q_norm,attn_k_norm,attn_norm,ffn_gate,ffn_up,ffn_down,ffn_norm}` ×6 + `fc.weight` + `enc.output_norm.weight` + `output_norm.weight`. **No `nextn`/`mtp` tensors, no `token_embd`, no `output.weight`** — a mask-token block drafter that consumes 8 target hidden layers through `fc.weight`. Only the first file parsed its tensors with a name lookup for the Q4_K_M one; in `dflash-draft-Ornith15.gguf` `blk.*.attn_output.weight [4096,2048]` is ggml type 30 = **BF16**.

### C.5 `/models/Qwen3.8-Flash-Next-Unsloth-IQ4_XS/`
- `mmproj-BF16.gguf` (865 MiB), `mmproj-F16.gguf` (862 MiB) — vision projectors.
- **`UD-IQ4_XS/` — target, 3 shards** `…-00001-of-00003.gguf` (10.4 MiB), `-00002-` (46.4 GiB), `-00003-` (40.8 GiB) = **87.25 GiB**, 1224 tensors, **arch `qwen4exp`**, `file_type=30`. Geometry: `block_count=48`, `embedding_length=2560`, `head_count=24`, `head_count_kv=2`, `full_attention_interval=4` → 36 GDN (`attn_qkv [2560,10240]`, `attn_gate [2560,6144]`, `ssm.*`) + 12 QSA (`attn_q [2560,12288]`, `attn_k/v [2560,512]`), `hyper_connection.count=4`, `hyper_connection.low_rank=320` (`hc_* [10240,…]`), `expert_count=512`, `expert(_shared)_feed_forward_length=640`, `context_length=262144`, `vocab=248320`, `attention.indexer.{head_count=4,key_length=128,top_k=2048}`, `attention.compress_ratios=[0,0,0,4,…]`, `ple.layers=[1]`, `ple.ngram_size=3`, `ple.heads_per_ngram=8`, `ple.conv_kernel=4`, `ple.eos_token_id=248044`, `ple.image_token_id=248056`, `ple.layer_multipliers=[23703573157769,20109073645365,8052911324071]`, `ple.head_offsets`/`head_vocab_sizes` (16 heads, ~2e8 rows each), `embedding_length_per_layer_input=160`. Tensors: `token_embd [2560,248320] Q8_0`, `output.weight [2560,248320] Q6_K`, `output_hc_{norm F32,down,up Q8_0}`, `blk.1.ple_*` (6 tensors), and **`per_layer_token_embd.weight [160, 320001536] IQ4_NL`** (the ~45 GiB UVA table). **MTP tensors: 0. `nextn_predict_layers` absent.**
- **`MTP/` — three draft-only artifacts, `block_count=49`, `name=Ckpt_Q38`:**
  | file | size | tensors | shared-target key |
  |---|---|---|---|
  | `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` | 2.59 GiB | **34** | *absent* → self-contained |
  | `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` | 1.78 GiB | **32** | `qwen4exp.nextn_shared_target_tensors = True` |
  | `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | 2.60 GiB | **32** | `qwen4exp.nextn_shared_target_tensors = True` |
  All three: `nextn_predict_layers=1`, `hyper_connection.count=4`, `low_rank=320`, `indexer.*`, `full_attention_interval=4`, `ple.*` metadata copied from the target — but **zero `ple_*`/`per_layer_token_embd` tensors**, i.e. the draft block carries no PLE.
  Draft block (`blk.48`) tensor set: `nextn.eh_proj [5120,2560]` (Q4_K / Q4_K / Q8_0), `nextn.enorm [2560] F32`, `nextn.hnorm [10240] F32`, `nextn.hc_head_norm [10240] F32`, `nextn.hc_head_down [10240,320]`, `nextn.hc_head_up [320,10240]` (Q4_K/Q5_0 in the Q4_K variants, Q8_0 in the Q8_0 variant) **plus the full dense block**: `attn_q [2560,12288]`, `attn_k/v [2560,512]`, `attn_output [6144,2560]`, `attn_q_norm/attn_k_norm [256] F32`, `indexer.{q_proj [2560,512],k_proj [2560,128],q_norm,k_norm} BF16` (loaded-then-skipped), `ffn_gate_inp [2560,512] F32`, `ffn_{gate,up}_exps [2560,640,512]`, `ffn_down_exps [640,2560,512] Q8_0`, `ffn_{gate,up,down}_shexp`, `hc_attn_{norm,down,up,inject}`, `hc_ffn_{norm,down,up,inject}`.
  Only `mtp-…-Q4_K_M.gguf` additionally carries `token_embd.weight [2560,248320] Q4_K` and `output.weight [2560,248320] Q6_K`. **`nextn.embed_tokens` and `nextn.shared_head_head` do not exist in any artifact** — LM head and embedding are always shared.

### C.6 `/models/Qwen3.8-Flash-Next-AD-4.27/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64/`
33 shards `…-00001-of-00033.gguf` … `-00033-of-00033.gguf`, **88.03 GiB total**, 1224 tensors, **arch `qwen4exp`**, `general.name=Src`, `file_type=28` (Q4_K_M). Geometry byte-for-byte identical to C.5's target (48 layers, 2560, 24/2 heads, hc 4×320, 512 experts, PLE layer 1, indexer top_k 2048, ctx 262144). `per_layer_token_embd.weight [160,320001536] **Q5_1**` (vs IQ4_NL in Unsloth), `output.weight [2560,248320] Q8_0`, `token_embd Q8_0`, shard 1 = 661 MiB (metadata + small tensors), shard 2 = 35.8 GiB (the PLE table). **MTP tensors: 0, and there is no `MTP/` sibling directory — this target has no MTP path on disk.**

### C.7 Non-GGUF FreeToken models
`/models/Qwen3.6-35B-A3B-NVFP4-FT/` — **`model_type qwen3_5_moe`**, arch `Qwen3_5MoeForConditionalGeneration`, 3 safetensors shards, index with **124468 tensors, 135.2 GB**. Config carries `mtp_num_hidden_layers: 1` and `mtp_use_dedicated_embeddings: False` but no `mtp` sub-dict. Geometry: 40 layers, hidden 2048, vocab 248320, ctx 262144, 16 q-heads / 2 kv-heads, head_dim 256, 256 experts, top-8, GDN (`linear_key_head_dim 128`, `linear_num_value_heads 32`, `linear_conv_kernel_dim 4`), `full_attention_interval=4`.
**19 MTP tensors, all in `model-00003-of-00003.safetensors`:** `mtp.{fc.weight, norm.weight, pre_fc_norm_embedding.weight, pre_fc_norm_hidden.weight}` + `mtp.layers.0.{input_layernorm,post_attention_layernorm}`, `mtp.layers.0.self_attn.{q,k,v,o}_proj.weight`, `q_norm`,`k_norm`, `mtp.layers.0.mlp.gate.weight`, `…shared_expert.{gate,up,down}_proj.weight`, `…shared_expert_gate.weight`, and **stacked `mtp.layers.0.mlp.experts.{gate_up_proj,down_proj}`**. Dedicated `lm_head.weight` (+ its NVFP4 `weight_scale`/`weight_scale_2`/`input_scale`) and `model.language_model.embed_tokens.weight` — the MTP head has **no** dedicated embedding despite `mtp_use_dedicated_embeddings: False` being consistent with that. This is exactly the family `models/qwen3_5_moe/weight.py:58` throws away.

`/models/Qwen3.8-Flash-Next-NVFP4-Radix/` — **`model_type qwen4_exp`** / `qwen4_exp_text`, arch `Qwen4ExpForConditionalGeneration`, **206 shards, 296475 tensors** (dense/small tensors in `model-bf16-*.safetensors`, one shard per 128 experts per layer: `layer-<id>-experts-<lo>-<hi>.safetensors` + `.complete.json`). Geometry: 48 layers, hidden 2560, vocab 248320, ctx 262144, 24 q / 2 kv heads, head_dim 256, 512 experts top-10, `hc_count=4`, `hc_lowrank=320`, `ple_layer_ids=[2]` (one-based in HF → zero-based 1), `ple_embed_dim=2560`, `ple_conv_kernel_size=4`, `ple_embedding_dtype=float8_e4m3fn`, `ngram_size=3`, `heads_per_ngram=8`, `ngram_vocab_size_base=20000000`, `make_ngram_vocab_size_divisible_by=128`, `split_ngram_parts=128`, `indexer_{n_heads=4,kv_heads=1,head_dim=128,budget=2048,compress_ratio=4}`.
**MTP config sub-dict (read by nothing in FreeToken):** `text_config.mtp = {"hybrid": true, "layer_types": ["full_attention"], "mtp_use_hidden_state_from_layer": null, "num_hidden_layers": 1, "rope_theta": 10000000}`.
**31 MTP tensors**, all BF16 in `model-bf16-000{10,11,12}.safetensors`: `mtp.{pre_fc_norm_embedding.weight, pre_fc_norm_hidden.weight, fc_embedding.weight, fc_hidden.weight}`, `mtp.hyper_connection_mixer.{hc_norm, input_mix_weight_down, input_mix_weight_up}.weight`, `mtp.layers.0.{attn,mlp}_hyper_connection.{hc_norm, input_mix_weight_down, input_mix_weight_up, block_inject_weight}.weight`, `mtp.layers.0.self_attn.{q,k,v,o}_proj.weight` + `q_norm`,`k_norm` + `mtp.layers.0.self_attn.indexer.{index_qk_proj, q_layernorm, k_layernorm}.weight`, `mtp.layers.0.mlp.{gate.weight, shared_expert.{gate,up,down}_proj.weight, shared_expert_gate.weight, experts.{gate_up_proj,down_proj}}`. **No `mtp.embed_tokens`, no `mtp.lm_head`** → shares. `hf_quant_config`/`quantization_config.ignore` is empty for the MTP group; the GGUF-side quant groups in `tests/models/qwen4_exp/common.py:294` show `mtp.layers.0.mlp.experts` as `FP8_PB_WO`.

---

## Synthesis — what it takes to add qwen4exp MTP to FreeToken-next (10 lines)

1. **Weight loading**: un-drop `mtp.*` in `models/qwen4_exp/weight.py:89` and `_EXPERT_KEY_RE` (`:42`) so `mtp.layers.0.mlp.experts.{gate_up_proj,down_proj}` reach the expert banks; map HF→engine: `mtp.pre_fc_norm_{embedding,hidden}` → `enorm`/`hnorm`, `mtp.fc_embedding`+`mtp.fc_hidden` → one fused `eh_proj [2*n_embd, n_embd]` (concat on dim 0, `e` repeated over the hc streams), `mtp.hyper_connection_mixer.*` → `nextn.hc_head_{norm,down,up}`.
2. **Config**: add `num_hidden_layers=1`, `hybrid`, `layer_types=["full_attention"]`, `mtp_use_hidden_state_from_layer` from `text_config.mtp` to `Qwen4ExpArgs` (`models/qwen4_exp/config.py`), and assert ≤1 draft block like `qwen4exp.cpp:56-61`.
3. **Model**: add `Qwen4ExpMTPDecoderLayer` by *reusing* `hc.py:GatedResidual`, `attention.py:Qwen4ExpAttention` minus the indexer (dense path), `moe.py:Qwen4ExpMoE`; **reuse `model.embed_tokens` and `lm_head`** — no dedicated tensors exist on disk.
4. **Handoff**: expose the pre-collapse wide residual `R [T, hc_count*hidden]` from `Qwen4ExpModel.forward` (`model.py:150-168`, currently returned only after `hyper_connection_mixer.mix`) as a graph output, at width `hc*hidden=10240`.
5. **Engine**: the hard part. `engine/engine.py:987-1005` produces 1 logit row + 1 token per request and `req.complete_one()`; you need N+1 output rows per request, a prefix-match accept step, and an accepted-length-driven `complete_n`. CUDA graphs (`engine/graph.py:79-99,140-204`) capture only a bs ladder — add a verify-length axis (or pad to `n_max+1` and mask), and mirror `llama-graph.h:959` by keying graphs on the nextn-hook flag.
6. **KV**: implement rollback for the trunk only (`kvcache/*` paged pool + `linear_state_pool`), and give the draft its **own** single-layer dense KV pool with no GDN/PLE state — `llama-model.cpp:2624-2632` is the blueprint. `kvcache/qsa_pool.py:49-51` already reserves ring headroom for `num_speculative_tokens` but is never passed one.
7. **GDN verify kernel is already there**: `kernel/fla/fused_sigmoid_gating_recurrent.py` accepts `disable_state_update`, `intermediate_states_buffer`, `intermediate_state_indices`, `retrieve_parent_token` — the multi-token verify path just has to pass them (`:263-280`); this is the single largest already-reusable asset.
8. **Carry/state**: port the `pending_h`/`verify_h` lifecycle (`speculative.cpp:2607-2638, 2808-2838, 3111-3175`) including the "stale carry ⇒ drop draft seq, keep newest target row" recovery, plus the never-`accept()`-twice rule; FreeToken's checkpoint/scheduler (`scheduler/scheduler.py:642-692`, `feat/decode-token-checkpoint`) has a comparable "restore ⇒ invalidate auxiliary state" notion to hang it on.
9. **PLE interaction**: the draft carries **no** PLE tensors (verified in all three `MTP/*.gguf` and in `mtp.layers.0.*`), so drafting does not touch the disk-backed table — but `model.py:152` `assert len(ple_layers)==1` and the graph-row sizing (`DiskRowTable(max_graph_rows=…, max_extend_tokens=…)`, `ple_disk.py:1-4`) must be widened to cover verify batches, and `forward_host_ctx` (`engine.py:991`) must stay outside captured regions.
10. **Flags & data**: add `--spec-type/--num-speculative-tokens/--spec-draft-p-min/--spec-draft-n-min` to `server/args.py` (nothing exists today); quant coverage is ready — the on-disk artifacts are `blk.N.nextn.*` at Q4_K/Q8_0 with `enorm/hnorm/hc_head_norm` in F32, and both NVFP4 checkpoints ship the MTP block in BF16, so `layers/quantization` needs no new scheme — and the existing tests that *assert MTP is never loaded* (`tests/models/qwen4_exp/test_weight.py:234-236`, `test_weight_ckpt.py:213-216`) must be inverted rather than duplicated.