# LESSONS — freetoken-next

**Formato:** `sintoma -> causa -> fix` | **Fonte:** Hardware real (RTX 5080), benchmarks live, serve crashes.

---

## CUDA / Kernel / JIT

- **CPU crash `malloc()` em Qwen4ExpModel.forward sem GPU** -> `VocabParallelEmbedding` usa kernel JIT CUDA-only sem guarda de device -> **fix:** `@requires_cuda` em todo teste end-to-end (padrão: `test_decoder_stack_prefill_and_decode`).

- **ptxas host-RAM OOM (70 GiB) ao fundir Turbo4 dequant + QSA attention** -> JIT não otimiza unrolled decode + pipelining junto -> **fix:** Split kernel: decompressão separada (`kernel/triton/qsa/decompress.py`) + attention denso; `block_n=32`, `num_stages=1` para kernels comprimidos.

- **Triton `fp4_quantization_120f` não compila com nvcc 13.3** -> `quantization.cu:488` alignment error -> **workaround:** test skip isolado; não bloqueia suite principal.

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
- **Decode eager (--no-graph) em 16K OOM com mem-ratio 0.9 (faltam 17MiB) -> sem captura de CUDA graph o probe de transient-peak mede 0 e o ledger de VRAM reserva menos, mas FLA chunk_delta_h.py aloca torch.empty_like(u) por passo que o graph absorvia -> fix: mem-ratio 0.85 cabe os transients eager; números eager com mem-ratio reduzido não são comparáveis a runs com CUDA graph (cache de experts encolhe, TG cai)** (2026-09-20)