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