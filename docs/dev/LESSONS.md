# LESSONS — freetoken-next

**Formato:** `sintoma -> causa -> fix` | **Fonte:** Hardware real (RTX 5080), benchmarks live, serve crashes.

---

## CUDA / Kernel / JIT

- **CPU crash `malloc()` em Qwen4ExpModel.forward sem GPU** -> `VocabParallelEmbedding` usa kernel JIT CUDA-only sem guarda de device -> **fix:** `@requires_cuda` em todo teste end-to-end (padrão: `test_decoder_stack_prefill_and_decode`).

- **ptxas host-RAM OOM (70 GiB) ao fundir Turbo4 dequant + QSA attention** -> JIT não otimiza unrolled decode + pipelining junto -> **fix:** Split kernel: decompressão separada (`kernel/triton/qsa/decompress.py`) + attention denso; `block_n=32`, `num_stages=1` para kernels comprimidos.

- **Triton `fp4_quantization_120f` não compila com nvcc 13.3** -> `quantization.cu:488` alignment error -> **workaround:** test skip isolado; não bloqueia suite principal.

- **CUDA graph capture crash `cudaErrorStreamCaptureInvalidated` em `qsa_sparse.py` (turbo4/qsa_sparse backend)** -> `valid = indices[indices >= 0]` é boolean-mask com shape dinâmica, operação inválida dentro de `torch.cuda.graph()` capture -> **fix:** guarda `capturing = torch.cuda.is_current_stream_capturing()`, pula seleção de páginas (`selected_pages=None`, decompress cai para todas as páginas) quando `capturing=True`. Commitado em `ba7f976`. **Reteste confirmado (2026-09-20):** decode via CUDA graph completa 2 repeats de 256 tokens sem hang em `replay()`, TG 33.10 tok/s.

- **`_measure_graph_capture()` em `memory_planner.py` nunca executava (`NameError: name 'Req' not defined`)** -> caminho de código morto: `cuda_graph_max_bs > 0` nunca foi exercitado em certificação antes desta sessão -> **fix:** import local `from freetoken.core import Req` (mesmo padrão já usado alhures no arquivo). Fase D agora mede `graph_capture_peak`/`graph_pool_size` de verdade sob CUDA graph mode.

- **Após o fix acima, serving real com `--moe-cache-auto` + CUDA graph mode trava no primeiro decode (0% GPU util após captura bem-sucedida)** -> reproduzido 1x em GPU, confirmado pré-existente via `git diff --stat` (não toca `graph.py`/`scheduler/`/backends de atenção) -> **NÃO investigado** (fora do escopo desta sessão, orientação do advisor). O planejador em si funciona sob graph mode (Fase D completa, plano é construído, prefill roda); o hang é no replay de decode, fora de `memory_planner.py`. Bloqueia certificação de "graphs + serving", não bloqueia "graphs + planner".

- **Verificação obrigatória em testes de MTP/graph:** não confiar só em exit code ou sha1 — sempre `grep -c "Traceback" <log>` deve ser 0. (Achado ao investigar o page-leak do MTP: o processo pai retornava 0 mesmo com traceback no scheduler background.)

- **`--spec-mtp 3` crasha em `spec.py:355` (`pool.recurrent_states[li, slot].copy_(rec_s)`: `output with shape [48,128,128] doesn't match the broadcast shape [1,48,128,128]`)** -> `models/qwen4_exp/gdn.py` capturava o checkpoint com `pool.recurrent_states[li, fla.cache_indices]` (indexação por tensor preserva a dim de batch) enquanto o restore em `spec.py` usa `pool.recurrent_states[li, slot]` com `slot` int (sem essa dim) -> **fix:** `.squeeze(0)` na captura (`fla.cache_indices` é sempre batch=1 nesse caminho de verify).

- **GGUF Unsloth-IQ4_XS falha ao carregar: `Shape/dtype mismatch model.layers.N.self_attn.indexer.index_qk_proj.qweight: expected [768,5120], got [640,5120]`** -> `models/qwen4_exp/gguf.py` setava `index_kv_heads=num_kv_heads` (KV heads da atenção principal/GQA), mas o indexer sempre usa 1 KV head compartilhado independente do GQA (`(index_n_heads + 1) * head_dim` esperado, não `(index_n_heads + num_kv_heads) * head_dim`) -> **fix:** ler `attention.indexer.head_count_kv` da metadata GGUF com default `1`.

- **GGUF Unsloth-IQ4_XS falha ao carregar (após fix acima): `PLE row source holds 320001536 rows but the hash addresses 320005046; incomplete checkpoint?`** -> `gguf.py` tinha duas chamadas a `derive_ngram_hash_constants` com `ple_layer_index` inconsistente: init do módulo usava o índice local entre camadas PLE (`ple.ple_index`, 0-based), mas o gerador de pesos (`iter_gguf_weights`, que sobrescreve via state_dict e é o valor que realmente conta) usava o índice absoluto da camada no modelo (`ple_layer_id`) -> gera primos/offsets diferentes para a mesma camada -> **fix:** `iter_gguf_weights` agora usa `enumerate(qwen4_args.ple_layer_ids)` para obter a mesma posição local que `Qwen4ExpPLE.ple_index` usa.

- **Comparação de TG entre versões usando números de boots "smoke" não certificados** -> stats.json de smoke test pode ter prompt de poucas dezenas de tokens mesmo citando `kv_reserve_tokens` alto (isso é reserva de VRAM, não profundidade real usada) -> **fix:** antes de comparar tok/s entre rodadas, sempre conferir `prompt_tokens_total`/`completion_tokens_total` no stats.json real, não assumir a partir de flags de configuração (`--max-seq-len-override`, `kv_reserve_tokens`).

---

## Processos / Monitoramento / Higiene

- **Proibido timeout/sleep/polling** -> tudo event-driven (Bash bloqueante, Monitor com condição real).

- **Subagent Opus "No transcript found"** -> transcripts não reaproveitáveis entre turnos -> **fix:** novo Agent com prompt autocontido (repetir contexto).

- **Múltiplos `tail -F` acumulados** -> logs duplicados -> **fix:** `TaskStop` no Monitor antigo antes de novo, ou reusar mesmo Monitor.

- **Zumbis `tail`/`python` pós-benchmark** -> degradam ambiente -> **regra:** `killall tail` + `manage_subagents kill` ao fim de cada análise.

- **Pre-Flight Check OBRIGATÓRIO** antes de qualquer benchmark:
  1. `free -h` — RAM livre, cache limpo
  2. `nvidia-smi` — VRAM livre
  3. `ps aux | grep -E '(python|tail)'` — matar órfãos
  4. `du -sh /tmp/*` — verificar lixo tmpfs

---

## MTP / Speculative Decoding

- **`ft serve --spec-mtp N` crasha em cascata** (KeyError peso MTP -> AttributeError export) -> wiring só exercitado em serve real -> **regra:** smoke-test `ft serve` real antes de declarar "implementado".

- **MTP k=1 OOM em 16K RTX 5080** -> `--spec-mtp 1` aloca draft weights (~126 MB extra) violando `--mem-ratio 0.9` -> **fix temporário:** `--mem-ratio 0.99` + `--no-graph`; **cura real:** TurboKV no QSA (4-bit KV).

- **MTP k=1 `--cuda-graph-max-bs 0` não é o gargalo** -> braço sem MTP usa mesma flag e roda 24.6 tok/s -> **causa real:** `spec.py` roda draft+verify como `Batch(phase="prefill")` a cada token + ≥2 host syncs/token (`tok_prev.item()`, `copy_done_event.synchronize()`).

- **Non-determinismo sequential requests same-server** -> 2 fases estáveis (req 1-2 idênticos, 3-6 idênticos mas diferentes) -> **não generaliza** (--decode 64 reverte) -> **próximo:** log logits top-2 em `spec.py:195` para discriminar numerics vs state bug.

- **MTP draft acceptance ~0%** -> prefill warm-up passava residual não-deslocado `h_t` com `x_t` (esperado `h_{t-1}` + `x_t` -> `x_{t+1}`) -> **fix:** carry shift 1 passo no residual (`res[0]=carry`, `res[1:]=last[:-1]`, `carry=last[-1]`). Aceitação 90.9%.

- **Repetição/gagueira streaming detokenização** -> `DetokenizeManager` assumia 1 msg/uid/lote -> **fix:** processar sequencialmente cada token por uid, isolando buffer.

- **k=2/k=3 corrigidos por inspeção, não testados live** -> draft chain per-step position bug (draft step i≥1 não via próprio KV prévio) -> **validar live** antes de confiar.

- **Prefill-window MTP warm-up AUSENTE** (EXP-025) -> draft head KV nunca populado sobre prompt original -> aceitação baixa nos primeiros tokens pós-prefill.

- **Batch detokenization com múltiplos tokens por UID em MTP (EXP-048)** -> quando spec-decode aceita $m \ge 2$ tokens, o scheduler despacha múltiplos `DetokenizeMsg` com o mesmo `uid` no mesmo lote -> `DetokenizeManager.detokenize()` acumulava todos os tokens no histórico antes de calcular `batch_decode`, mas os offsets de leitura assumiam 1 mensagem por UID -> resultado: fatia de caracteres duplicada (`--0` em vez de `-0`), alterando o texto e o SHA1 mesmo com os tokens do modelo 100% idênticos -> **fix:** detectar repetição de UID no lote e processar sequencialmente para manter os invariantes de offsets progressivos.

- **Bisecção de commit não serve quando o bug sobrevive a TODOS os reverts testados** (EXP-048) -> sha1 idêntico entre árvore 100% revertida e árvore com MoE quebrado (TG 0.47 tok/s) -> **causa está fora do range bisectado** -> **fix:** ao ver conteúdo invariante a reverts de código funcional, trocar de eixo (aqui: comprimento de decode em vez de commit) para achar o ponto exato de divergência antes de continuar revertendo commits.
- **KL 0.97 verify vs decode na posição 4098 (turbo3)** -> não é corrupção: GEMM bf16 do cuBLAS dá resultado diferente com 1 linha e com 2+ linhas; na 4098 os logits turbo3 estão quase empatados (margem 0.19) e amplificam isso. Prova: GEMMs linha a linha na janela de verify tornam todas as linhas bitwise iguais ao k0 -> **sem fix** (ft-campaign2/A2b-outlier-4098.md, patch de diagnóstico rowwise-gemm-debug.patch).

---

## VRAM / RAM / Cache / Paging

- **Flash-Next offload NVFP4 precisa ~63.46 GiB RAM host** -> host tem ~60GB -> earlyoom mata worker ~150/192 experts -> `--moe-cache-size` não ajuda (é VRAM) -> **medir `free -h` antes de retry**.

- **earlyoom "8-9GB usado" pós-crash ≠ pico real ~69GB** -> earlyoom recalcula "user mem total" descontando tmpfs/shm -> **fix:** limpar `/tmp` + `systemctl mask tmp.mount` (permanente após reboot).

- **`--moe-cache-auto` decidia split KV/experts ANTES de `--num-tokens` explícito** -> pedido 16576 tokens virava 8256 silenciosamente -> **fix:** `--num-tokens` vira piso obrigatório, nunca sugestão.

- **Page leak em `allocate_paged` não-idempotente** -> replay GDN rebobinava cached_len/device_len e realocava página cruzando limite -> **fix:** `_prepare_batch(..., skip_alloc=True)` em replay paths.

- **LinearStatePool (GDN snapshot) vazava 1 slot/request** -> `_spec_eligible_req` desliga spec no último token, cleanup só no branch spec -> **fix:** mover `free_spec_snapshot_slot` para `_free_req_resources` (caminho compartilhado por abort/decode/spec).

- **Output sha1 diferente entre repeats mesmo prompt/servidor** -> vazamento estado entre requests sequenciais -> **validar:** 2 processos servidor SEPARADOS com 1 request cada + comparar sha1.

- **Dummy-page brick (`num_pages + 1`)** -> ambas fórmulas de budget agora o prezam -> guards hold.

- **VRAM Ledger single source of truth** -> ceiling, reserve, overhead, expert/KV split, context rows owned by one account.

---

## GGUF / MoE Geometry

- **A7 first pass: expert rows 276/308 B -> achou layout privado** -> offset-derived table (1194/1194 tensors) mostra llama.cpp byte counts sob ids llama.cpp -> **não há GPU row reader faltando** -- `--moe-strategy offload` já dequantiza todo `BLOCK_SHAPE`. Gap é só CPU dot kernels (Q3_K/IQ3_S/IQ4_XS).

- **`_host_ram_fits_parallel` compara checkpoint total (125.9 GiB) vs bank real (63.46 GiB)** -> parallel path permanentemente unreachable -> **fix junto com** transient budget undercount (1 vs 3 shards) se revisitado.

---

## Benchmarking / Validação

- **Prefix cache faking PP measurement** (`#new-token: 64` de 16384) -> **harness usa `--cache-type naive`** sempre.

- **Atribuir regressão a flag sem comparar baseline MESMA flag** -> `--cuda-graph-max-bs 0` presente em ambos braços -> **regra:** comparar baseline com MESMA config.

- **Hipótese "estabiliza depois N requests" (autotune/JIT)** -> confirmada em --decode 4, DESMENTIDA em --decode 64 -> **rodar teste confirmação com parâmetro diferente antes de documentar**.

- **Suíte verde não prova bug fixado** -> commit page-leak (EXP-033) testou helper novo, não os 3 hunks reais -> **todo bug fix precisa teste exercendo `check_integrity()` original falhando antes**.

- **Fix em cache/paginação compartilhada -> rodar suite COMPLETA do subsistema** (não só arquivos tocados pelo diff) -> EXP-033 fix quebrou 11 testes scheduler não-relacionados.

- **Non-determinismo sequential persistia após EXP-039 zerar ring/scratch** -> carrier restante era o **slab KV do draft slot MTP** (camada 48), sujo por páginas pré-alocadas do verify e recicladas via radix -> **fix:** `clear_mtp_slot()` em `QSAKVCache.free_req` (todos os tiers: cmp_k, pending_ring, `_k_codes`/`_k_norm`/`_v_codes`/`_v_norm`/BF16 `_kv_buffer`); determinismo só é provado com repeats sequenciais bit-identical, não probe de um carrier por vez.

---

## Performance / Compression

- **KV compression é capacity lever, não 16K speed lever** -> Triton uncompressed já custa 9.1% TG em 16K -> quantizer não recupera lá; **compression compra:** 256K deixa 5427 expert slots vs 3183 BF16.

- **Packed layout só vence se reader sai do address business** -> byte-per-element + gather centroid book/element -> 4x menos bytes, ITL dobrou (6.79→15.98ms) -> **fix:** load packed row once, split in registers.

- **In-SRAM FP4 dequant vs DRAM workspace** -> `_ws_k`/`_ws_v` em DRAM penaliza bandwidth O(N) -> **fix:** dequant direto no loop SRAM do kernel `attend.py`.

---

## Upstream / Prior Art

- **Upstream PRs já têm seams exatos necessários:** KV-dtype plumbing (#354, #408, #113), speculative decoding (#69), NVMe expert tier (#337), GGUF mixed-quant (#494).

- **FreeToken-Kai (191 commits, +31.8k lines)** já mergeou nossa base exata e tem GGUF/KV-quant/host-bank/long-context/VRAM-accounting -> **port-with-provenance > reinvenção**.

- **TurboQuant solicitado upstream (#141)** -> nosso Turbo3/Turbo4 backend é diferencial.

- **`qwen4_exp` MTP tensors existem nos checkpoints** mas upstream loader **dropa `mtp.*`** (#421) -> Phase 9 greenfield na nossa base.

- **Auditoria Cruzada LTO (`llama-turbo-optimal`) vs FreeToken para maximização de TG**:
  - *Taxa de aceitação != maior TG*: aceitações menores (50-60%) com profundidade maior (`n_max` 3-4) podem superar amplamente TG de aceitações altas (80%+) com `n_max` 1-2 se a sobrecarga de despacho/verify for enxuta.
  - *Sobrescrita silenciosa no workspace QSA*: bloco `else` em `qsa_sparse.py` sobrepunha a descompressão Turbo4 com tensores brutos comprimidos durante fases não-decode (prefill/verify) -> **fix**: estruturar `if compressed:` estrito englobando decompress e atenção.
  - *Gargalos de PCIe/Host no MoE*: `_SMALL_BANK_FEAT_BYTES` a 256 KiB forçava cópias síncronas de bancos pequenos quantizados -> **fix**: reduzir piso para 64 KiB e elevar paralelismo de fetch (`hybrid_max_fetch=4`, `hybrid_fetch_fraction=0.5`).
  - *VRAM over-reservation*: reservas estáticas de autotune e capture peak aprisionavam ~320 MiB de headroom que agora alimentam o pool de páginas KV / slots MoE.

---

## Referências Cruzadas

| Arquivo | Uso |
|---------|-----|
| `CONTEXT.md` | Visão geral projeto, estado, próximos passos |
| `STATE.md` | Snapshot executável, handoffs, comandos |
| `ROADMAP.md` | Fases 1-18, gates, dependências |
| `DECISIONS.md` | D-001 a D-023, rationale imutável |
| `PERFORMANCE.md` | Tabelas PP/TG/VRAM/RSS por config |
| `ARCHITECTURE.md` | Design, seams, file:line |
| `EXPERIMENTS.md` | EXP-000 a EXP-045, setup/result/verdict |
| `QA.md` | Gates, validações, checklists |
| `AGENTS.md` | Instruções para agentes IA |
- **Cache leak em replay MTP -> _prepare_batch skip_alloc=False -> skip_alloc=True em replay paths (2026-09-19)** (2026-09-19)
- **Teste auto-doc -> script funciona -> doc atualizada** (2026-09-19)
- **Documentação EXP-050 afirma prefill-window MTP warm-up implementado (45% accept) mas não existe código em scheduler/spec.py para popular draft KV no prefill** (2026-09-19)
- **STATE.md afirma _spec_eligible_reqs() multi-request removida trava single-request mas scheduler/spec.py mantém _spec_eligible_req() singular com len(running)!=1 check** (2026-09-19)
- **STATE.md/EXPERIMENTS.md afirmavam EXP-048 "resolvido" com sha1 17f277f43565 sem log em disco** -> processo em background interrompido sem registro na sessão anterior -> **fix:** nunca marcar experimento como validado sem artefato (log/json) rastreável; live re-run confirmou k=0 sha1 `dec18d678a16` (baseline atual, corpus/config podem ter mudado desde a claim original — comparar sempre contra sha1 do MESMO run, não de sessões antigas).
- **Flash-Next earlyoom trigger (~8.1GB threshold) durante serve boot/prefill -> Linux page cache retém ~10GB lendo safetensors + glibc retém heap arenas -> posix_fadvise(POSIX_FADV_DONTNEED) pós-load de pesos em engine.py + malloc_trim(0) em scheduler.py liberam ~1.5GB de host RAM, garantindo MemAvailable acima do trigger de 10% do earlyoom** (2026-09-19)
- **MTP k>=1 e k>=2 no Flash-Next 16K exigem --kv-format turbo4 na RTX 5080 16GB -> sem turbo4 o KV em BF16 estoura VRAM ao carregar draft layer MTP -> com turbo4 e --memory-ratio 0.86 a VRAM estabiliza em 13.88 GiB e determinismo bit-identical é 100% certificado (sha1 match k=0, k=1, k=2)** (2026-09-19)
- **CUDA 13.3 + FlashInfer JIT: alignas(64) CUtensorMap falha compilação C++ -> Em CUDA 13.3 o tipo CUtensorMap possui alinhamento default de 128 bytes; alignas(64) tenta reduzir alinhamento e gera erro de compilação no nvcc -> fix: alterar para alignas(128) CUtensorMap permite build JIT limpo de fp4_quantization_120f.so** (2026-09-19)
- **OffloadMoeCache default hybrid_fetch_fraction deve ser 0.0 -> FREETOKEN_HYBRID_FETCH_FRACTION padrão 0.0 preserva teste test_hybrid_fixed_cap_unchanged e política canônica de capped fetch quando profile não é especificado** (2026-09-19)
- **freetoken_kernel_cache/__init__.py deletado (uncommitted) -> _load_prebuilt() sempre retorna None -> todo kernel JIT recompila do zero em cada start (compile storm, ~70GiB RSS) -> fix: git checkout do arquivo + rebuild/install de ambos os wheels via scripts/build-release-wheels.sh (FREETOKEN_BUILD_NO_STAMP=1 se tree sujo)** (2026-09-20)
- **iostat -x: %util é coluna distinta de w_await -> confundir w_await (98ms) com %util invalidou inteiramente hipótese de disk-bound MoE thrashing -> fix: sempre conferir o header de colunas do iostat -x antes de concluir I/O-bound** (2026-09-20)
- **WIP do MoE offload trava indefinidamente após token 1 com CUDA graph decode habilitado -> causa raiz não confirmada (py-spy `-s --native` localiza em torch/cuda/graphs.py replay(), engine/graph.py:214, mas não vê além do pybind/driver; FUSED_COPY=0 e SMALL_BANK_FEAT_BYTES=262144 não resolvem) -> fix: usar --no-graph como baseline eager válido; detalhes do profiling em PERFORMANCE.md; causa raiz pendente de cuda-gdb/nsys** (2026-09-20)
- **Decode eager (--no-graph) em 16K OOM com mem-ratio 0.9 (faltam 17MiB) -> sem captura de CUDA graph o probe de transient-peak mede 0 e o ledger de VRAM reserva menos, mas FLA chunk_delta_h.py aloca torch.empty_like(u) por passo que o graph absorvia -> fix: mem-ratio 0.85 cabe os transients eager; números eager com mem-ratio reduzido não são comparáveis a runs com CUDA graph (cache de experts encolhe, TG cai)** (2026-09-20)- **GGUF Unsloth-IQ4_XS após fix de indexer/PLE ainda crasha em request real: `AssertionError: PLE needs the ple_ngram_ctx slot state` (`ple.py:358`), reproduzido com `cache-type=radix` e `naive`, com e sem `--no-graph`** -> `_ngram_context_pool()` não encontra `PLE_NGRAM_STATE` registrado no pool para o path de carregamento GGUF -> -> **causa raiz:** `parse_config` (path safetensors, `config.py:299`) passa `slot_states=ple_slot_states(qwen4_args)` ao `ModelConfig`, mas o `ModelConfig(...)` construído em `gguf.py` (path GGUF) omitia esse argumento por completo, deixando `slot_states` vazio e o pool nunca registrava `PLE_NGRAM_STATE`/`PLE_CONV_STATE` -> **fix:** adicionar `slot_states=ple_slot_states(qwen4_args)` ao `ModelConfig` em `gguf.py`. Verificado: benchmark 4096 tok prompt / 32 decode roda end-to-end (TG 31.36 tok/s, sem crash).
- **`cert_matrix.py --contexts 16384` marca `gguf-flash-unsloth-ud` como FAIL por "stall-timeout" após o fix do PLE_NGRAM_STATE** -> não é hang: log do servidor mostra prefill chunk de 8192/16384 tokens levando ~41s (PP ~194 tok/s, IQ4_XS sem kernel MMQ dequantiza no prefill) -> 2 chunks facilmente excedem os 90s fixos do watchdog de `bench_pp_tg.py` -> **não confundir "TTFT prefill timed out" com deadlock de GPU** sem antes checar o log do servidor por progresso real de tokens processados.
- **mtp_sha1_divergence -> verify_forward prefill attention numerics differ from decode -> always replay accepted tokens through decode path in spec.py run_spec_step** (2026-09-20)
- **mtp_low_acceptance -> draft head KV cold start with only last chunk residual -> warmup_mtp_draft_kv now runs full prefill context without spec_logits_indices** (2026-09-20)
- **mtp_tg_regression -> token-a-token replay in decode path doubles forward passes -> need batched multi-token decode replay (blocked on kernel support)** (2026-09-20)
- **mtp_optimal_k -> K=4 achieves 27.8 tok/s (K=1: 24.5, K=2: 24.3, K=3: 18.8, K=4: 27.8, K=5: 14.8, K=6: 13.2) with SHA1 matching k=0 baseline** (2026-09-20)
- **flash_next_tg_ceiling assumed PCIe-bound -> `--moe-strategy hybrid` (48.7% PCIe/51.3% CPU, auto-tuned) moved only +3% TG (29.61->30.51, bit-identical) -> PCIe expert-streaming is NOT the dominant decode-step cost on Flash-Next; real gap vs 35B-A3B (6.3ms/token vs 33ms/token, same GPU) is expert VRAM residency, not transfer bandwidth. Do not chase PCIe-hiding kernel work expecting >single-digit % gains.** (2026-09-20)
- **`--ple-backend pinned` on Flash-Next -> backend worker killed during load (exitcode=-15) -> RSS already ~70 GiB with `disk` PLE backend at mem-ratio 0.98, pinning the full n-gram table needs more RAM than free -> retest only at a lower mem-ratio / after freeing host RAM elsewhere.** (2026-09-20)
- **`FREETOKEN_DEBUG_SPEC_TIMING=1` on MTP k=4 shows draft_chain (~11ms/4 forwards) << verify_forward (~140ms/5 positions) -> MTP k=4's -7% TG is fully explained by verify cost vs ~40-50% acceptance (no bug); `gdn_replay` fallback never fires, `zero_replay_gdn` covers 100% of restores -> the batched-replay "fix" from an earlier STATE.md next-step was already done; don't re-propose it.** (2026-09-20)
- **Per-layer `torch.cuda.Event` timing (opt-in `FREETOKEN_DEBUG_LAYER_TIMING=1` in `models/qwen4_exp/model.py`) shows MoE costs ~2.5x the GDN/QSA mixer per layer call -> corrects the earlier "expert VRAM residency" guess above: the Flash-Next vs 35B-A3B gap is more likely dominated by MoE grouped-GEMV/dequant compute cost than by PCIe/VRAM residency (hybrid test already ruled out PCIe). Treat the VRAM-residency line above as superseded.** (2026-09-20)
- **`--quant-backend moe.nvfp4=b12x` (flashinfer SM12x W4A16) on Flash-Next -> crash `force_tile_config fc2 tile (tile_k=32, tile_n=512) does not fit problem N/K=2560/640` -> flashinfer's tile-config table doesn't cover this checkpoint's MLP geometry; not fixable without an upstream flashinfer patch. `nvfp4_marlin` is sm_80-99 only, incompatible with this sm_120 GPU by design -> `triton` native kernel is the only viable NVFP4 MoE backend here.** (2026-09-20)

- **Planejador `--moe-cache-auto` OOM determinístico em contexto 16K, mesmo após Fase I validar o plano** -> `phase_h_construct_final_pools` construía pools só para a validação, mas nunca eram liberados antes de `engine.py` reconstruir os pools de produção logo após `plan()` retornar -> **fix:** `plan()` desanexa (`attach_offload_moe_cache(model, None)`) e libera (`del` + `gc.collect()` + `empty_cache()`) os pools de validação a cada iteração do replan e antes do retorno.

- **`AssertionError: No active batch in context` na Fase I de validação do planejador** -> `_run_validation_prefill` chamava `model.forward()` sem `global_ctx.forward_batch(batch)`, diferente do `_run_probe_prefill` (que já usava o padrão correto) -> **fix:** envolver o forward de validação no mesmo context manager.

- **`_cleanup_probe_artifacts` não recuperava memória do probe (driver_free não subia após cleanup)** -> `attach_offload_moe_cache(model, cache)` grava `layer.offload_cache` direto nas camadas do modelo real, e `global_ctx.linear_state_pool`/`attn_backend` ficavam apontando pro probe -- referências fortes fora do escopo do planner, nunca desanexadas -> **fix:** desanexar explicitamente (`attach_offload_moe_cache(model, None)`, restaurar originais salvos de `global_ctx`) antes de deletar os atributos `_probe_*`; driver_free pós-cleanup subiu de 2.47 GiB para 5.31 GiB.

- **Turbo3 16K OOM real de serving (não do planejador) no primeiro forward, dentro do kernel GDN compilado via torch._dynamo** -> `RuntimeCalibration.triton_autotune_peak`/`backend_workspace_peak`/`allocator_fragmentation` são constantes hardcoded (128/128/64 MiB) nunca recalibradas por medição real, insuficientes pro workspace real do kv-format turbo3 -> **não corrigido**; precisa virar medição real por kv-format antes de certificar Turbo3/MTP/CUDA graphs.

- **Hipótese "validação de 1 chunk só" testada e DESCARTADA por A/B real em GPU** -> implementado 2º chunk de continuação (`initial_state` preenchido) em `_run_validation_prefill`, gated por `FT_VALIDATE_CONTINUATION`; rodando Turbo3 16K com o fix off vs on produziu planos finais **byte-idênticos** (experts=1317, kv_pages=259, mesma trajetória de replan) e o **mesmo OOM real byte-a-byte** (79.38 MiB livres, faltando 96 MiB) em ambos -> a especialização Triton do chunk de continuação não era a causa raiz; **não reverter o código** (é validação estritamente melhor, mede ~30 MiB a mais de fato), mas não atribuir a ele a correção do OOM do Turbo3.

- **Gap real confirmado entre Fase I e serving: validação relata `free_after=1.59 GiB` (medição real do driver pós-forward) para o mesmo plano que falha no serving real com só ~79 MiB livres** -> causa raiz ainda NÃO isolada. Descartado por leitura de código: `engine.py` usa fielmente `plan.expert_slots`/`plan.kv_pages`/`plan.prefill_chunk` (não é bug de "plano ignorado"). Candidatos a investigar: pools de produção construídos por `engine.py` após `plan()` retornar podem incluir algo que os pools de validação da Fase H não têm (scheduler/radix-cache/fila de requests, ou `kv_reserve_tokens` — default 8192, não usado em lugar nenhum de `memory_planner.py`, só em `engine.py:884`).

- **Bug real encontrado (não confirmado se é a causa do gap acima): `RuntimeCalibration.backend_workspace_peak` contado duas vezes na mesma soma** -> `_other_transient_bytes()` soma `backend_workspace_peak` e `_semi_persistent_bytes()` soma `backend_workspace_peak` de novo -> `_residual_for_chunk` e o invariant check de `_solve_experts_and_kv` usam os dois juntos, então todo replan que aumenta `backend_workspace_peak` por X bytes desconta 2X do residual disponível -> **avaliado via advisor: não corrigido de propósito** — só torna o planejador mais conservador (nunca pode causar um OOM), então não é a causa raiz; deixado como está.

- **Hipótese `_warmup_prefill()` pulado em `qsa_sparse` (turbo3) testada por raciocínio, sem GPU, e DESCARTADA** -> `_warmup_prefill` roda forwards de 80/128 tokens; autotune do Triton é keyed por shape/config, então não pré-compila o launch real de 8192 tokens; além disso a Fase I do planejador já roda um forward real de 8192 tokens (`_run_validation_prefill`) no mesmo processo, antes do plano ser aceito -> qualquer custo de JIT já foi pago ali, não pode reaparecer "de surpresa" no primeiro request real. Não perseguir essa hipótese de novo.

- **Causa raiz real #1 do OOM de serving do Turbo3 (confirmada + corrigida + validada em GPU 2026-09-21): a única medição física real da Fase D (`peak_alloc_delta`, de uma probe real via `_run_probe_prefill`) era descartada** -> `phase_d_runtime_calibration` calcula `measured_transient = peak_alloc_delta` (delta real do allocator numa probe real) mas nunca usava esse valor; `triton_autotune_peak`/`backend_workspace_peak` ficavam hardcoded em `128 * _MIB` cada (comentário "Will be refined", nunca implementado) -> confirmado nos logs: probe real mediu `peak_alloc=0.82 GiB` em chunk=4096, mas a fórmula pura da Fase E prevê só `gdn_peak≈0.42 GiB` pra esse chunk (linear a partir de `0.83 GiB` em chunk=8192) — ~0.40 GiB de custo real de backend/autotune não explicado pelo GDN, contra só 0.25 GiB reservados pelos hardcodes -> **fix:** `backend_workspace_peak = max(0, peak_alloc_delta - gdn_peak_at(probe_chunk))` (residual medido, específico do `attention_backend` real em uso, não um chute igual pra todo backend); `triton_autotune_peak` zerado (já incluído no residual medido, evita contar o mesmo custo duas vezes). Medição real em turbo3: `backend_workspace_peak` passou de 0.125 GiB (placeholder) para 0.407 GiB (medido).

- **Causa raiz real #2, exposta só depois do fix acima apertar o orçamento (confirmada + corrigida + validada em GPU 2026-09-21): `RuntimeError: Joint solver invariant violated`** -> `StaticCostModel.fixed_overhead_bytes()` somava `kv_fixed_bytes` + `dummy_page_bytes` como custo "fixo", mas todo cálculo de `kv_bytes` (`kv_bytes_for_context` e as duas cópias inline em `_solve_experts_and_kv`/`plan()`) já cobra `kv_fixed_bytes` + a página dummy (via `pool_pages(pages) = pages+1`) de novo, por completo -> os mesmos bytes de KV eram cobrados duas vezes contra o mesmo orçamento; com a reserva de transiente antiga (256 MiB hardcoded) sempre sobrava folga suficiente pra esconder o bug, com a reserva medida (~0.4 GiB) o orçamento ficou justo o bastante pra estourar por ~0.09 GiB (exatamente a ordem de grandeza de `kv_fixed_bytes + dummy_page_bytes`) -> **fix:** removidos `kv_fixed_bytes`/`dummy_page_bytes` de `fixed_overhead_bytes()` — esses termos só devem ser cobrados uma vez, dentro do `kv_bytes` real de cada candidato.
- **Turbo3 16K CERTIFICADO após os dois fixes acima**: 3/3 execuções reais de GPU (`--kv-format turbo3 --moe-cache-auto --max-seq-len-override 16384`, prompt de 15800 tokens), todas `exit code 0`, `kv_pages=261` idêntico nas 3, **saída bit-idêntica** (sha1 `a23eca87ec47`). PP≈1790 tok/s, TG≈19.3-20.7 tok/s, TTFT≈8820 ms.

- **`--spec-mtp 1` sobre Turbo4 16K: request completa e sha1 bate com o baseline (`098ee9a58aec`), mas o scheduler crasha logo depois, ao ficar idle** -> `CacheManager.check_integrity()` (scheduler/cache.py:595, ramo `naive`/não-hybrid/não-swa) acusa `free_pages(258) + cache_pages(0) != num_pages(259)` — 1 página de KV não devolvida após o rollback especulativo. Confirmado **não relacionado** aos fixes desta sessão: `git diff --stat` mostra que nenhuma mudança tocou `scheduler/cache.py` nem `engine/spec.py`; o bug é pré-existente no caminho de rollback do MTP, não no planejador de VRAM. Acceptance rate real medido via `FREETOKEN_DEBUG_SPEC_TIMING=1`: 32/95 = 33.7% (não é "zero aceitação", spec decode está funcionando, só mais lento que o k=4 registrado em 2026-09-20 por causa do contexto de 16K aqui vs. contexto menor daquele teste) -> **não corrigido, fora do escopo desta sessão** (bug em `scheduler/cache.py`/`engine/spec.py`, não em `memory_planner.py`); bloqueia certificação de MTP k=1..6 até ser investigado/corrigido separadamente. Todo teste de MTP/CUDA-graph daqui em diante precisa checar `grep -c "Traceback" <log> == 0`, não só o `exit code`/sha1 do bench — o cliente reporta sucesso mesmo com o scheduler morto no fundo.

- **Causa raiz real do page-leak do MTP acima (confirmada + corrigida + validada em GPU 2026-09-22, 4 repeats sem crash): `CacheManager.allocate_paged` recalcula `first_page = div_ceil(req.cached_len, page_size)` do zero a cada chamada, sem memória do que já foi alocado antes** -> instrumentação de page-trace (alloc/free por página física, gated por env var, removida após o diagnóstico) isolou exatamente 1 página (`15808`) alocada uma única vez e nunca liberada em nenhum ponto do log inteiro; o gatilho: um rollback especulativo (draft rejeitado, só o token de correção committed) deixa `keep_cached` EXATAMENTE num múltiplo de `page_size`, depois que essa página já tinha sido alocada 1 token dentro dela (pro draft rejeitado) — `div_ceil(cached_len, page_size)` nesse caso aponta pra essa MESMA página como "ainda não alocada", então a rodada seguinte aloca uma página física NOVA pro mesmo intervalo lógico e `_write_page_table` sobrescreve a entrada da tabela, órfã a física antiga pra sempre (nunca mais referenciada nem devolvida ao free-list) -> **fix:** novo campo `Req.alloc_page_bound` (índice de página, alta-marca-d'água de posse real): `allocate_paged` usa `first_page = max(div_ceil(cached_len, page_size), req.alloc_page_bound)` e sempre seta `alloc_page_bound = last_page`; `free_spec_reject` seta `alloc_page_bound = first` (o início do intervalo devolvido) sempre que roda, senão o piso ficaria acima de páginas já liberadas e elas nunca mais seriam realocadas pra essa request. Cobre também de graça o caso de replay/`_prepare_batch` chamando `allocate_paged` duas vezes sobre o mesmo intervalo (idempotente agora). Testes novos/atualizados em `test_spec_reject_frees_pages.py` (`test_allocate_paged_twice_over_the_same_range_is_idempotent`, `test_rollback_to_an_exact_page_boundary_after_a_speculative_page_does_not_orphan_it`); suite completa 23/23.
- **GGUF Unsloth-IQ4_XS OOM no offload cache VRAM planner mesmo em min_experts minimo -> checkpoint dynamic-quant: ffn_gate_exps/ffn_up_exps/ffn_down_exps variam de tipo ggml por camada (IQ3_S/IQ4_NL na maioria, IQ4_XS/Q8_0 num subconjunto), offload_cache.py:446-454 aloca um pool GPU cheio (cache_size slots) POR geometria distinta em vez de dividir entre elas -> nao e bug de gate_up layout (gate/up sao tensores separados que concatenam limpo); e planner territory (fora de escopo, ver decisao anterior); requer decisao do usuario: reduzir min_experts viavel para N geometrias, ou aceitar que este checkpoint nao cabe em GPU de 16GB** (2026-09-22)
- **FreeToken planner: budget de VRAM double-conta backend_workspace_peak (transient E semi-persistent) em _other_transient_bytes/_semi_persistent_bytes (memory_planner.py) -> residual ficava ~0.4GB menor que o real -> fix: remover backend_workspace_peak de _other_transient_bytes, manter so em _semi_persistent (2 call-sites)** (2026-09-22)
- **ft serve sem --max-seq-len-override usa o context_length nativo do GGUF (262144 para Qwen3.8-Flash-Next) em vez do contexto pretendido -> planner tenta reservar paginas KV para 256K, nao 16K, e estoura VRAM -> sempre passar --max-seq-len-override explicito ao testar contexto especifico em modelos GGUF** (2026-09-22)
- **PYTORCH_ALLOC_CONF=expandable_segments:False com este modelo (GDN+QSA+MoE mixed-quant GGUF) trava driver CUDA em compute 100% permanente, zumbi nao reciclavel segurando VRAM, systemd PID1 em D-state -> nao e so mais lento, e um hang real de driver -> NUNCA desabilitar expandable_segments para testar alocador neste modelo; exigiu reboot da maquina** (2026-09-22)
- **Tentativa de re-sincronizar physical_budget do planner com take_physical_snapshot logo apos phase_d_runtime_calibration (sem empty_cache antes) mediu VRAM NO MEIO do probe (pools transientes ainda vivos), nao depois -> deu budget=0.06GiB (falso), pior que nao fazer nada -> fix descartado/revertido; qualquer re-snapshot do planner PRECISA seguir o padrao synchronize()+empty_cache() antes de ler driver_free, como o Phase H ja faz** (2026-09-22)
- **memory_planner.py phase_b_build_static_model orçava gdn_slots via _linear_pool_min_slots (piso sem cache de snapshot), mas phase_h_construct_final_pools SEMPRE constrói a pool final via _linear_pool_num_slots (inclui cache cross-request, maior) -> todo plano aprovado pelo solver estava sub-provisionado em (num_slots-min_slots)*bytes_por_slot (0.44-0.88GB tipico) -> um plano 'valido' podia OOM na construcao final -> fix: phase_b usa _linear_pool_num_slots, igual ao que phase_h de fato constrói. Teste: tests/engine/test_memory_planner_gdn_slots.py** (2026-09-22)
- **GGUF 16K OOM com ~2.4 GiB de VRAM "não-PyTorch" surgindo na primeira prefill (probe do planner)** -> serving GGUF usava `_LazyGGUFBank` (experts em host pageable/mmap, não pinned); `copy_missing` lê via ponteiro host e o driver em modo HMM migra as páginas para VRAM fora do caching allocator -> **fix:** voltar ao load eager pinned (`PinPipeline`); nunca deixar GPU ler host pageable. Diagnóstico: wrap de funções medindo `(driver_used - reserved)` delta.
- **CUDA graph decode trava no 1º token (GPU 0%, CPU 100% em cuGraphLaunch/sched_yield)** -> PLE disk `wait-sync` grava no graph um cuStreamWaitValue64 cujo flag o host só sinaliza após `replay()` retornar; graph grande (turbo KV / MoE GGUF) faz cuGraphLaunch bloquear antes -> deadlock -> **fix:** `FREETOKEN_PLE_SYNC` auto = launch-gating. Diagnóstico: py-spy --native (servidor com prctl PR_SET_PTRACER_ANY).
- **GGUF + `--spec-mtp` crasha "Cannot pack tensors on meta"** -> gguf.py troca `embed_tokens` por GGUFEmbedding mas `model.mtp._embed_ref` segue no embedding meta antigo -> **fix:** re-apontar `_embed_ref` após a troca.
- **Servidores de certificação morrem com exitcode -15 durante load** -> earlyoom (RAM <10%) ao rodar pytest paralelo junto com servidor de ~70 GB RSS; e `make ci` (`--basetemp=/models/desenvolvimento/tmp`) apaga logs ali -> **fix:** nunca rodar CI junto com certificação; logs de cert em /models/desenvolvimento/certlogs.
- **GGUF Flash-Next "certificado" gerava texto sem sentido (só TG era medido)** -> loader qwen4_exp GGUF: normas plus-one carregadas com +1 (llama.cpp dobra o +1), V-heads do GDN em ordem tiled, output_gate `silu` em vez de `sigmoid` -> **fix:** `_plus_one_norm`, un-tile V-heads (bytes Q8_0 em ssm_out), gate `sigmoid`; toda certificação checa texto/sha contra FTW, nunca só TG. (2026-09-22)
- **MTP GGUF 0% de aceitação** -> `GGUFLMHead` só pontuava a última linha no verify (ignorava `spec_logits_indices`) -> **fix:** `select_head_rows` compartilhado por GGUFLMHead/gemma4. (2026-09-22)
- **MTP k=1 aceitação 55 % vs LTO 87 % (mesmo GGUF MTP)** -> `model.py:221` `moe_layer_id=first_k_dense_replace` + `iter_gguf_mtp_weights` pula `blk.48.ffn_*_exps`: router do MTP roda sobre experts da camada 0 (cosseno blk.48 vs blk.0 ≈ 0, pesos independentes) -> **fix pendente:** carregar blk.48 exps como banco próprio. Ver `relatorio-estudo-mtp-lto.md`. (2026-09-22)
- **Bench `--spec-mtp 1` morre no boot: `--spec-mtp > 0 requires overlap scheduling disabled`** -> aceitação é decidida no host -> **fix:** `FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1` (custo em k=0: 35.36 -> 35.08 TG). (2026-09-22)
- **MTP com banco blk.48 próprio: planner recusa 16K (falta 0.7 GiB)** -> planner reservava piso de 2×num_experts slots para prefill overlap que o cache desliga depois (geometria GGUF mista) -> **fix:** `engine.py` desliga overlap antes do planner quando os bancos têm geometria mista. Custo restante: gate_up Q8_0 do MTP é geometria nova = pool extra de cache_size slots (slots 1181 -> 738). (2026-09-22)
- **Planner committed 0.67 GiB under budget with geometry pools -> expert_bytes_for_slots(0) priced the pools decode floors instead of 0 (residual ledger asks for 0 slots) -> expert_cache_bytes returns 0 for rows<=0 (2026-09-22)** (2026-09-22)
- **MoE decode stats never print on server stop -> uvicorn SIGINT terminates the scheduler worker, so engine.shutdown never runs -> send SIGINT to the worker child first (pgrep -P) (2026-09-22)** (2026-09-22)
- **Mixed GGUF expert geometry cost one row of EVERY geometry per cache slot (5.8 MB target, 9.3 MB with MTP Q8_0 vs ~2.3 MB real), so slots fell 1181 -> 738 -> per-geometry slot caches sized cache_size -> byte-budget geometry pools (cache_budget.expert_pools), k1 22.3 -> 28.4, k0 35.1 -> 41.4 (2026-09-23)** (2026-09-23)
- **torch.cuda.graph fails with "must be captured on a non-default stream" in unit tests -> GraphRunner captures on the caller stream -> build the runner under torch.cuda.stream(side_stream)** (2026-09-23)
- **k1 bench printed "sha SAME" yet k1 text differed from k0 -> tgclient SAME only compares the 3 reps with each other -> diff the out-*.txt against the k0 run explicitly** (2026-09-23)
- **disk PLE staged 1 row for a 2-token verify window, draft position read a stale pinned row -> host fill sliced req.input_ids, which holds committed tokens only -> Batch.spec_host_ids carries the window (28f7ec6); pinned vs disk PLE sha is the oracle** (2026-09-23)
- **GDN verify checkpoints cloned every verify but never restored -> spec.py hard-codes checkpoints = None, so replay always runs -> the verify graph skips them; zero-replay must add static checkpoint buffers** (2026-09-23)
- **Campaign dir under /models/desenvolvimento/tmp vanished mid-run -> make ci runs pytest --basetemp=/models/desenvolvimento/tmp, which wipes that whole directory -> keep campaign artifacts outside the pytest basetemp (e.g. /models/desenvolvimento/ft-campaign)** (2026-09-23)
- **Verify-graph check failed only on 4-row windows, QSA cmp_k tensor -> non-closing rows all write the per-slot scratch row (racing, never read) -> compare the slab up to cmp_scratch_base only** (2026-09-23)
- **Two-token verify cost ~2x dense GEMV time -> vendored mmvq launched one block row per input vector, re-reading weights -> keep up to 4 partial sums per block (bitwise-equal per vector)** (2026-09-23)
- **Freshly booted server hung forever after GGUF load, 0% GPU -> an interrupted run left ~/.cache/torch_extensions/.../freetoken_gguf_kernels/lock and the next JIT load waits on it -> when no nvcc/ninja runs, delete the stale lock; never kill a server during its first kernel JIT** (2026-09-23)
- **Bench PP anchors (3153/3112) looked like cold prefill -> tgclient repeats the same prompt and FreeToken ignores cache_prompt:false, so every measured rep hits the radix prefix cache -> measure cold PP with a unique prefix per request (ft-campaign/coldclient.py); true cold PP was 377 tok/s at 4K** (2026-09-23)
- **Cold GGUF prefill 10.8 s for 4K tokens -> MoE ran the per-row GEMV (moe_vec_q, 6.4 s) and dense Q8_0 the vendored MMQ (2.1 s, ~9x slower than dequant+cuBLAS on SM120) -> dequantize routed experts 16 at a time into the bf16 fused-MoE kernel, dequant+matmul for dense; PP 377 -> 1385** (2026-09-23)
- **MTP draft-step CUDA graph (1-row, captured after verify graphs) hits illegal memory access on its first replay, even without check mode -> root cause not found (not the QSA verify buffers, which are per-size) -> shelved; patch in /models/desenvolvimento/ft-campaign/draft-graph.patch, expected gain only ~2 ms/cycle** (2026-09-23)
- **`cache_prompt:false` requests still matched the radix prefix cache -> the server never read the `cache_prompt` request field, so cold-bench requests silently reused a warm prefix -> skip the prefix-cache match when `cache_prompt:false` is set (86eae99); `tests/scheduler/test_cache_prompt_cold.py`** (2026-09-23)
- **Cold prefill logits differed 2.8-3.8 across identical requests (non-deterministic) -> `_fused_experts_dequant`'s bf16 `index_add_` combine had no fixed write order (atomic order + per-expert bf16 rounding varied per run) -> write each (token,slot) row once, sum top-k in fixed order fp32 (211efb6); DET2 confirmed bitwise-equal reruns** (2026-09-23)
- **`--moe-strategy hybrid` produced garbage output ("!!!!" from the first decode token) -> the CPU hybrid executor applied one dominant (gate_up, down) format to the whole checkpoint, but GGUF layers mix formats per layer/bank (e.g. IQ4_XS gate_up + Q8_0 down, Q8_0 MTP layer read with the wrong block stride) -> NaN -> decode each GGUF layer with its own expert formats instead of the checkpoint's dominant pair (7bbf2af)** (2026-09-23)
- **Verify-window KL vs decode logits looked like a spec-decode correctness bug (up to KL~1) -> cuBLAS bf16 linears are not row-independent (M=1 row0 differs from M>=2 row0 by 2.5e-4..2.4e-3, M>=2 rows agree among themselves) -> verify windows of any width share one arithmetic, decode uses another, both are "correct"; deferred replay adds no new error beyond ordinary verify (`FREETOKEN_SPEC_DEFER_REPLAY=0` kept only as a fallback, not required); method: teacher-forced dumps via `FREETOKEN_DEBUG_LOGIT_DUMP` + `ft-campaign2/kl_compare.py`, comparing only up to the first committed-token divergence on matched positions** (2026-09-23)
- **GPU left busy / port 8091 held between campaign runs, skewing slot counts and TG on the next run -> a worker's killed process left orphaned scheduler children and a bench server holding the port instead of exiting -> bench.sh now refuses to start if GPU >500 MiB used or the port is busy, and always kills the server's process-group children on exit** (2026-09-23)
- **A worker's worktree edits and scratch data vanished mid-campaign -> a worker ran `git worktree` cleanup / edited the shared `wt/` by mistake instead of a pre-allocated per-worker scratch dir -> pre-allocate one worktree/scratch dir per worker before dispatch, never let a worker create or delete `wt/` itself; keep evidence outside the pytest basetemp** (2026-09-23)
- **MTP draft-graph v2 still hits illegal memory access in `qsa_forward` on first replay -> the pre-allocated-capture-pool-output hypothesis was tested and did not fix it, true cause still unknown -> shelved pending `compute-sanitizer` memcheck on replay to name the faulting kernel/address (QSA draft-slot addressing suspected)** (2026-09-23)
- **Servidor trava no carregamento (GPU ~1.7 GB, 0% uso, log parado após gguf reader)** -> lock órfão `~/.cache/torch_extensions/py312_cu130/freetoken_gguf_kernels/lock` de um build JIT morto; o próximo boot espera o lock para sempre -> **fix:** matar processos, `rm` do lock, pré-compilar com `python -c "from freetoken.kernel.gguf import _module; _module()"`.
- **`ft tune` mediu MTP ligado igual a desligado** -> requests com temperature 0 herdavam top_p 0.95 do modelo e `is_greedy` exigia top_p == 1, então o MTP nunca rodava -> **fix:** 5de83aa (temperature 0 ou top_k 1 é greedy).
- **Híbrido GGUF 3.5x mais lento que offload** -> fração de busca procurada pela tag "gguf", que o bench bw não grava; caía em limite fixo de 1 expert -> **fix:** d52f287 (chave pelo par gate_up+down). Justo: 37.05 vs 40.48 offload.
- **Auditoria estimou 6-9 ms/ciclo de ganho com overlap hit/miss** -> raciocínio invertido: overlap esconde no máximo o tempo de GEMV dos hits, não o da cópia; trace mostra cópia 19.3 ms vs GEMV 4.0 ms por ciclo -> **fix:** medir kernel a kernel (torch.profiler) antes de codar; NO-GO (ft-campaign2/B2-verdict.md).
- **`ft tune` perdeu os candidatos MTP (committed_tg null)** -> earlyoom matou o scheduler (RSS 73 GB) porque 4 processos órfãos de rodadas anteriores seguravam 5.6 GB -> **fix:** matar órfãos (`ppid 1`, `spawn_main`, venv do repo) antes de tunar; conferir `journalctl | grep earlyoom`.
- **Perfil do `ft tune` desligava o draft graph mesmo com ganho** -> empate dentro da margem de 3% favorece o primeiro candidato, e `draft_graph=False` vinha primeiro -> **fix:** af2651d (padrão do engine primeiro).
- **Dois `ft tune` 16K discordaram (MTP +11% vs +1.5%)** -> cada candidato recebia um nonce de prompt por tempo, então decodificava texto diferente (aceitação e misses de expert mudam com o texto) -> **fix:** f90a5e2 (prompts pareados por rep); tune8/tune9 reproduziram dentro de 0.6%.
- **Perfil novo do `ft tune` ignorado no boot (spec_mtp=0)** -> `tune_cli` gravava `schema=1` fixo após o bump para 2 -> **fix:** 14e0478 (`Profile.schema` usa `SCHEMA_VERSION`).
- **Boot OOM na validação do VRAM ledger** -> agente rodou pytest com testes CUDA na GPU durante o boot (260 MiB) -> **fix:** agentes sem GPU usam `CUDA_VISIBLE_DEVICES=""` em todo comando.
- **Pacote `freetoken/debug` sem `__init__.py` importado por scheduler/spec** -> `[tool.setuptools.packages.find]` ignora diretórios sem `__init__.py`; só funcionava em install editável -> **fix:** adicionado `python/freetoken/debug/__init__.py`.
- **`/models/outros/cuda-13.3/bin/nsys` falha: "Nsight Systems ... hasn't been installed with CUDA Toolkit 13.3"** -> o binário é um wrapper shell sem o Nsight instalado -> **fix:** usar `/models/outros/nsys-ncu/extracted/opt/nvidia/nsight-systems/2026.1.3/target-linux-x64/nsys` envolvendo o `bench_pp_tg.py`.
- **Trace nsys do servidor sem os últimos ~100 passos de decode** -> bench encerra o servidor com SIGTERM e o CUPTI não esvazia o buffer final -> **fix:** tratar a janela capturada como amostra em regime estável (contar `cudaGraphLaunch`), nunca como run completo.
- **Q8_0 -> Q6_K dense re-quant looked like -23% bytes/time on paper** -> Q6_K MMVQ reaches only ~467 GB/s vs ~552 GB/s for Q8_0 (graph-replay microbench, campaign 11) -> **fix:** measure kernel time per quant type before converting; use the time ratio (Q6_K 0.94, Q5_K 0.71, Q4_K 0.65), not the bpw ratio.
- **AD-4.27 byte model predicted +9.5..10.5% TG, A/B gave +4.97%** -> LRU proxy from a short k1 trace plus bytes-proportional kernel time overstates savings (IQ2_S vec cost, compulsory misses) -> **fix:** discount offline byte ceilings ~2x; only authorize conversions whose ceiling clears 10%+.

- **Chat GGUF com raciocínio devolvia resposta vazia/cortada (UD 12/20 tarefas)** -> `load_gguf_tokenizer` não registrava tokens CONTROL/USER_DEFINED; `<think>`/`<tool_call>` viravam texto ("<th","ink",">") no prompt renderizado; benchmarks em texto puro não detectam -> **fix:** `b5d4243` registra tipos 3/4 como AddedToken; comparar ids com `llama-tokenize` ao qualificar modelo novo.
- **ISTA IQ3_XXS não carregava** -> tipo ggml Q2_0 (42) desconhecido + `ssm_out` em quant de bloco 256 corta cabeça V de 128 -> **fix:** Q2_0 genérico + gather da entrada do out_proj (sem cópia densa, sem VRAM).
- **turbo3/turbo4 parecia alavanca de velocidade nos históricos de k1** -> históricos k1>k0 mediam com turbo3, que derruba k0 11-13% -> **fix:** comparar k0/k1 sempre com o mesmo kv-format, registrado no ledger.
- **"Aceitação MTP caiu de 75% para 55%" parecia regressão da engine** -> aceitação depende do conteúdo do prompt (probe-prompt4k ~60%, prompt16k ~86%, mesma engine e cabeça) -> **fix:** comparar aceitação sempre no mesmo prompt e reportar 4K e 16K separadamente.
- **MTP k1/k2 com aceitação >75% ainda perdia ou ganhava pouco** -> com experts em RAM, cada token verificado roteia experts novos (só ~18% de sobreposição) e o token do draft acha o cache mais frio (acerto 66% vs 75.6%); verify de 2 tokens busca 2.2x bytes pela PCIe -> **fix (pendente):** atacar o custo de bytes do verify (mais slots residentes, overlap de gather, residência ciente de verify) antes de buscar mais aceitação.
- **Rastreador MoE (`FREETOKEN_MOE_TRACE`) vazio com CUDA graph** -> replays do grafo não passam pelo Python -> **fix:** rastrear com `--no-graph`.
