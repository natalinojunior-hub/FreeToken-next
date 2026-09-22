# Relatório Definitivo — Planejador de VRAM (FreeToken)

Fonte verificada em código: `python/freetoken/engine/memory_planner.py`, `benchmarks/cert_matrix.py`, `tests/engine/test_memory_planner_ledger.py`.

## 1. Equação canônica de VRAM

O orçamento (`budget`) é o `driver_free` medido **uma única vez**, via `take_physical_snapshot`, após a limpeza dos artefatos de probe (`_cleanup_probe_artifacts`). Esse valor já exclui: pesos do modelo, contexto CUDA, módulos carregados, crescimento não-PyTorch e qualquer alocação persistente que tenha sobrevivido à limpeza — logo nenhum desses termos entra de novo no ledger (teste `test_budget_owners_are_not_double_counted`).

O ledger (`MemoryPlanner.ledger`, linhas 681-705) soma, por dono: `gdn_state + page_table + expert_aux + quant_tables + staging + attn_backend (medido na construção, Fase C) + experts(slots × bytes/slot, geometria deduplicada) + kv(pool_pages(pages) × kv_bytes_per_page + kv_fixed_bytes) + lazy_persistent (medido no warm-up) + graph_pool + transient`.

`transient = max(transient_at(chunk), graph_capture_peak)` — não somados, pois captura de grafo e prefill nunca coexistem (teste `test_graph_capture_and_prefill_peaks_are_not_summed`). `transient_at(chunk)` interpola linearmente entre dois pontos medidos (`chunk_lo`/`transient_lo`, `chunk_hi`/`transient_hi`) — cada ponto medido em `reserved` (`max_memory_reserved`, `_measure_prefill_transient`, linha 567), no pior passo: 1º chunk + chunk final no fim do contexto — nunca extrapolado acima do maior medido; chamada acima de `chunk_hi` dispara `assert` (`RuntimeCalibration.transient_at`, linha 205; teste `test_transient_is_linear_between_measured_points_and_floored_below`).

**Condição de viabilidade:** `soma(ledger) ≤ budget`. **Prioridade de alocação** (`phase_fg_solve_chunk_and_experts`, linhas 722-760): 1) contexto pedido é piso rígido de páginas KV; 2) maior chunk de prefill da escada `_CHUNK_LADDER` que ainda deixa resíduo ≥ 1 slot de expert; 3) resíduo vira slots de experts até `max_expert_slots`; sobra sub-slot vira páginas KV extras (testes `test_solve_funds_context_then_largest_chunk_then_experts_within_budget`, `test_solve_shrinks_chunk_before_failing_expert_floor`).

**Validação Fase I** roda o pior prefill real (chunk final no fim do contexto), medido em `reserved` (`max_memory_reserved`, linha 920), sem margem arbitrária — falha = dono não precificado (comentário l.1125 "failing means unpriced owner"), reportado com ledger e budget (linhas 1126-1130). **Inviável** → erro antes da alocação final: piso pré-Fase-C (linhas 1070-1074) e `infeasible` (linhas 707-719) reportam `required=`, `available=`, `shortfall=` e os 5 maiores donos (teste `test_infeasible_reports_required_available_shortfall_and_owners`).

## 2. Causas-raiz encontradas e corrigidas (com bytes)

a. Experts GGUF eram servidos de RAM host não-fixada (`_LazyGGUFBank`, commit wip `db07811`); GPU lia via HMM, driver migrava ~2,41 GiB para VRAM fora do alocador PyTorch. Fix: carregamento pinned (PinPipeline); crescimento não-PyTorch caiu de 2,41 GiB para 0,01 GiB. (Não localizado em `memory_planner.py`, não verificado neste ciclo.)

b. Probe media só 1 chunk de 4096 e o solver extrapolava constante para 8192 (transiente real 1,76 GiB vs 0,84 GiB medido). Fix: 2 pontos + interpolação linear — confirmado: `RuntimeCalibration.transient_at` (linha 205) interpola entre `chunk_lo`/`transient_lo` e `chunk_hi`/`transient_hi`, com `assert` contra extrapolação acima do medido (teste `test_transient_is_linear_between_measured_points_and_floored_below`).

c. `backend_workspace` (0,52 GiB) contado como semi-persistente e somado ao transiente (dupla contagem); `non_pytorch` medido mas nunca cobrado; Fase F/G resolvia contra um budget e Fase H re-resolvia contra outro (contraditório); laço de até 20 re-tentativas por OOM deixava o KV crescer de volta; margem arbitrária de 256 MiB na Fase I; termo fixo chutado de 128 MiB para backend de atenção (real medido: 0 MiB). Todos removidos/substituídos pelo ledger único — confirmado no código atual: `attn_fixed = 0` (linha 389), sem placeholder de 128 MiB; `_physical_budget_hint` não existe mais em `memory_planner.py` (ver item i). O laço de retentativas por OOM da Fase D (linhas 589-606) hoje é bounded por `c_hi <= _MIN_CHUNK` (256), sem contador fixo — consistente com "removido".

d. Validação usava tabela de páginas parcial; agora `_run_validation_prefill` mapeia o contexto completo (linha 965) e roda dois chunks: `cached_len=0` (primeiro) e `cached_len=max_seq_len - chunk` (final, linhas 1007-1010).

e. `turbo_kv.pack/unpack` criava tensor no host dentro de captura de CUDA graph. Fix: shifts por bit / `arange` no device — confirmado: `python/freetoken/kernel/triton/turbo_kv.py` usa `torch.arange(..., device=device)` (linhas 381, 514) e operadores de bit `>>`/`<<`/`|`/`&` (linhas 385, 471-520), sem criação de tensor em host.

f. MTP: slot de snapshot GDN (`spec.py:_spec_snapshot_slot`) alocava sem liberar snapshots do radix → crash quando lista livre esgotada. Fix: `ensure_mamba_slots(1)` antes de alloc — confirmado: `self.cache_manager.ensure_mamba_slots(1)` (linha 114 de `python/freetoken/scheduler/spec.py`); teste `test_spec_snapshot_slot_evicts_radix_snapshots_when_free_list_is_drained` em `tests/scheduler/test_spec_reject_frees_pages.py`.

g. CUDA graph travava no 1º decode (GPU 0%, CPU 100% em `cuGraphLaunch`/`sched_yield` no driver, pilha nativa via py-spy). Causa: em `python/freetoken/models/qwen4_exp/ple_disk.py` o modo "wait-sync" da tabela PLE em disco insere no graph uma espera (`cuStreamWaitValue64`) por um flag que o host só escreve DEPOIS de `graph.replay()` retornar; com grafo grande (descompressão turbo KV; MoE GGUF) `cuGraphLaunch` bloqueia antes de retornar → deadlock. Fatorial medido: FTW bf16 passa, FTW turbo trava, GGUF trava até em bf16; prefill overlap e planner inocentados. Fix: modo "auto" agora escolhe "launch-gating" (`FREETOKEN_PLE_SYNC=wait` ainda força o modo antigo). Resultado: todos os graphs passam.

h. GGUF + MTP: a cabeça MTP mantinha referência ao embedding original (meta, nunca carregado) depois que `gguf.py` trocava `embed_tokens` pelo `GGUFEmbedding` → "Cannot pack tensors on meta". Fix: re-apontar `model.mtp._embed_ref` para o novo embedding (`python/freetoken/models/qwen4_exp/gguf.py`).

i. Placeholder chutado de 128 MiB para backend de atenção removido também do piso pré-Fase-C (agora 0 até medido); guarda morta `_physical_budget_hint` removida.

j. Harness `benchmarks/cert_matrix.py`: 1 boot por configuração, 3 cargas no mesmo boot; watchdog por evento (servidor morto, `/health` erro, GPU ociosa ≥2 s sem token) aborta na hora, despeja pilhas Python (`faulthandler`) e nativas (py-spy; servidor faz `prctl PR_SET_PTRACER_ANY`).

k. CI: teste `tests/models/test_muse_glimmer.py::test_iter_weights_bf16_matches_model_state_dict` (2 casos, ~102 dos 111 s do pytest) marcado `@pytest.mark.slow`.

## 3. Reconciliação PLANEJADO|REAL (todas as 7 configurações certificadas)

Mecanismo confirmado: `log_reconciliation` (l.123-132) compara `planned`/`actual` por dono, `warning_rank0` se `|delta| > 128 MiB`; chamado na Fase C (l.527), Fase H (l.822), Fase I (l.921). Números do cenário GGUF 16K turbo3, mr=1 (execução, repassados pelo enunciado, não presentes no código-fonte): Fase C `kv` 87,4/87,4, `experts` 5675,0/5675,5, `gdn` 551,4/551,4 MiB — este `gdn` de Fase C usa `_linear_pool_min_slots` (linha 520), um piso de slots menor que o pool final; Fase H `gdn_state` 992,6/992,6 MiB usa `_linear_pool_num_slots` (linha 818), a mesma fórmula do dimensionamento completo (seção 4) — 551,4 × 9/5 ≈ 992,5, consistente com a razão de slots entre as duas fases, não uma divergência; `experts` 6772,3/6772,8, `kv` 87,7/87,7 MiB; Fase I transiente 1804/1904 MiB (+100 MiB = backend de atenção novo criado na validação recriando tensores lazy, equivalente ao termo `lazy_persistent` de 104 MiB contado à parte). Nenhuma discrepância inexplicada > 128 MiB — consistente com o limiar de warning do `log_reconciliation` confirmado em código (linha 129).

Transiente Fase I, PLANEJADO/REAL (MiB), nas 7 configurações certificadas:

| config | planejado | real | delta | lazy_persistent |
|---|---|---|---|---|
| gguf-t3 | 1804 | 1904 | +100 | 104 |
| gguf-t4 | 1824 | 1902 | +78 | 104 |
| ftw-t3 | 868 | 906 | +38 | — |
| ftw-t4 | 828 | 866 | +38 | — |
| ftw-t3-mtp1 | 808 | 1006 | +198 | 184 |
| gguf-t3-mtp1 | 1762 | 1900 | +138 | 184 |
| gguf-t4-mtp1 | 1702 | 1902 | +200 | 184 |

Explicação: a Fase I cria um backend de atenção novo que recria os tensores "lazy" dentro do pico medido; o ledger já os cobra à parte como `lazy_persistent`; delta ≤ lazy + ~16 MiB de arredondamento. Nenhuma discrepância inexplicada > 128 MiB. Fases C e H: deltas ≤ 0,5 MiB nas 7 configurações.

## 4. Desperdício estrutural remanescente (não corrigido)

Cache de experts usa um índice de slot GLOBAL para 4 geometrias GGUF (gate_up IQ3_S 47 camadas / IQ4_XS 1 camada; down IQ4_NL 43 / Q8_0 5): cada slot reserva linha nas 4 → 5,81 MB/slot vs ~2,33 MB usados por expert típico (~2,5× sobre-provisionado). Correção exige pools de slot por geometria no kernel LRU. Parcialmente confirmado em `python/freetoken/moe/offload_cache.py`: array global único `self.id_of_slot` (linha 214) e `expert_geometry` keyed por `(bank_idx, role, quant_type)` (linha 251); valores exatos em MB não recalculados — repassados do enunciado.

`gdn_num_slots` no cenário mr=1: `_linear_pool_num_slots` — teste `test_gdn_slots_budgeted_at_built_pool_size` confirma a fórmula para 1 requisição (ratio=2,0): 4 slots de trabalho + `max(4, 2)` snapshots do radix + 1 de padding = 9 slots total (Fase H mede `gdn_state` = 992,6 MiB = 0,97 GiB, consistente em todas as 7 configurações), alocados antecipadamente (não lazy).

Checkpoints MTP `(k+1)×0,108 GiB` são cobertos pela medição de transiente/lazy (ver seção 6), não por termo analítico separado.

## 5. Certificação GPU (RTX 5080 16 GB)

Contexto 16384, 3 cargas no mesmo boot (256 tok, 4096 tok, contexto cheio ~16256 tok, 64 tokens de decode); harness `benchmarks/cert_matrix.py` — ver detalhes do watchdog e captura de pilhas no item j (seção 2). GPU ociosa 2 s sem token → hang, confirmado em `benchmarks/bench_pp_tg.py`: `gpu_idle_timeout` default 2.0 (linha 300), via `nvidia-smi --query-gpu=utilization.gpu`; `cert_matrix.py` fixa `stall_timeout` em 30,0 s (linha 103).

Resultado FINAL (todos com CUDA graph padrão, planner automático, sem flags manuais de memória, contexto 16384, 3 cargas — 256 / 4096 / contexto cheio ~16256 tokens, 64 tokens de decode; PP da carga "short" inclui aquecimento):

| config | status | boot (s) | short PP/TG | 4k PP/TG | fill PP/TG | VRAM máx (GiB) |
|---|---|---|---|---|---|---|
| gguf-t3 | PASS | 169 | 144/40,8 | 388/28,6 | 511/28,0 | 14,82 |
| gguf-t4 | PASS | 168 | 144/29,1 | 388/28,7 | 510/28,1 | 14,82 |
| ftw-t3 | PASS | 61 | 207/32,5 | 1795/30,9 | 2295/26,5 | 14,83 |
| ftw-t4 | PASS | 56 | 207/29,0 | 1788/31,5 | 2285/26,8 | 14,87 |
| ftw-t3-mtp1 | PASS | 57 | 207/19,4 | 1795/16,4 | 2289/10,6 | 14,87 |
| gguf-t3-mtp1 | PASS | 169 | 145/10,0 | 388/10,0 | 511/11,5 | 14,97 |
| gguf-t4-mtp1 | PASS | 167 | 144/10,2 | 388/9,95 | 511/9,97 | 14,89 |

Nenhum OOM, nenhuma razão manual de memória, nenhuma margem arbitrária, nenhum retry. O bloqueio de CUDA graph descrito em versões anteriores deste relatório foi corrigido (causa raiz e fix: seção 2, item g).

## 6. MTP

O planner cobre MTP via medição (`lazy_persistent` sobe de 104 para 184 MiB com MTP; transiente medido, não extrapolado); todas as configurações MTP carregam e servem 16K sem OOM. PORÉM: MTP k=1 REDUZ a TG — FTW cai de 26-32 para 10,6-19,4 tok/s, GGUF cai de ~28 para ~10 tok/s. É uma regressão de desempenho do MTP, fora do escopo de VRAM, não investigada nesta sessão. Checkpoints GDN do MTP `(k+1)×0,108 GiB` são cobertos pela medição de transiente/lazy, não por termo analítico.

## 7. CI

`make ci` PASS (CPU-only, GPU oculta): format/lint OK, mypy 427 arquivos 0 erros, pytest 1548 passed / 516 skipped. Novos testes: `tests/engine/test_memory_planner_ledger.py` (8 testes: aritmética do ledger, transiente linear sem extrapolação, graph vs prefill não somados, contexto como piso, chunk encolhe antes de falhar, relatório de inviabilidade, preço exato do KV com página dummy, sem dupla contagem de donos já fora do budget); teste em `tests/scheduler/test_spec_reject_frees_pages.py` para o slot MTP. Incidente: rodar pytest em paralelo com o servidor (~70 GB RSS) acionou o earlyoom e matou workers — não rodar CI e certificação juntos. `make ci` usa `--basetemp=/models/desenvolvimento/tmp`, que APAGA essa pasta (logs de servidor em execução ali somem).

## 8. Bloqueadores/remanescentes

1. Desperdício estrutural do cache de experts GGUF por geometria (~2,5×) — exige pools de slot por geometria no kernel LRU.
2. Pools GDN alocados antecipadamente (9 slots, mr=1, 0,97 GiB; 4 são snapshots do radix) — alocação lazy de snapshots não implementada.
3. Regressão de TG com MTP (ver seção 6).
4. Contextos 128K/256K ainda não certificados (o ledger trata o contexto como piso rígido e rejeita antes da alocação, com required/available/shortfall, se não couber).

Nada commitado.
