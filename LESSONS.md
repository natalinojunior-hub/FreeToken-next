# LESSONS

sintoma -> causa -> fix

crash CPU (`malloc(): unsorted double linked list corrupted` / Fatal Python Abort) ao rodar
Qwen4ExpModel.forward em teste sem GPU -> VocabParallelEmbedding usa kernel JIT CUDA-only
(kernel/index.py -> index.cu), sem guarda de device -> qualquer teste que chame forward de
ponta a ponta precisa de `@requires_cuda` (padrão já usado em test_decoder_stack_prefill_and_decode).

regra do operador: proibido timeout/sleep/polling neste projeto -> tudo deve ser event-driven
(Bash bloqueante direto, Monitor com condição real, nunca `timeout N cmd` nem loops de sleep).

SendMessage a um subagent Opus falhou com "No transcript found" -> subagents lançados via Agent
ficam sem transcript reaproveitável entre turnos longos -> ao precisar de follow-up profundo,
lançar um novo Agent com prompt totalmente autocontido (repetir todo o contexto necessário)
em vez de tentar retomar pelo agentId antigo.

3 Monitors abertos e nunca fechados na mesma tail -F de log, cada retry de serve criou um novo
-> mensagens duplicadas/repetidas pro operador a cada linha -> ao reiniciar um processo que já
está sendo monitorado, ou chamar TaskStop no Monitor antigo antes de abrir um novo, ou reusar o
mesmo Monitor (ele sobrevive a um serve que morre e reinicia, desde que o arquivo de log seja o
mesmo). Nunca acumular monitores paralelos na mesma fonte.

`ft serve --spec-mtp N` crasha em cascata de bugs de integração nunca exercitados por teste
unitário (KeyError de peso do MTP -> AttributeError de export do pacote) -> qualquer wiring novo
que só é alcançado num serve real (carregamento de peso, registro de módulo em __init__.py)
precisa de um smoke-test de serve real antes de declarar "implementado", não só suíte unitária
verde. Suíte unitária verde != funciona no serve ao vivo.

Flash-Next em offload NVFP4 precisa de ~63.46 GiB de RAM do host pros expert banks; o host tem
~60GB disponíveis -> earlyoom mata o worker por volta de 150/192 experts carregados, com ou sem
MTP (não é bug introduzido pelo MTP) -> bloqueio pré-existente e já documentado (EXP-015);
`--moe-cache-size` não ajuda (é VRAM, não RAM do host). Não insistir em re-tentar sem antes medir
RAM disponível (`free -h`) e comparar com o footprint conhecido do bank.

regra do operador: no máximo 2 retries numa falha que se repete -> depois disso, medir a causa
de verdade (RSS trace, logs, free -h) em vez de tentar de novo às cegas. Aplicado em EXP-029:
o crash do earlyoom era uma corrida de timing numa margem de RAM já apertada (mesmo pico de RSS
do baseline, 68.4 GiB), não um vazamento do MTP -- só descobri medindo, não re-tentando.

`free -h` mostrando "8-9GB em uso" DEPOIS de um crash de earlyoom não contradiz um pico real de
~69GB DURANTE o carregamento -> são dois momentos diferentes; o operador tinha razão em duvidar
("não é falta de RAM") mas a causa raiz era outra: earlyoom mede um "user mem total" recalculado
que desconta tmpfs/shm, não o MemTotal físico fixo -> cada 1GB parado em /tmp custa ~0.9GB da
margem de 10% do SIGTERM (`earlyoom --dryrun -r 1` prova isso imprimindo os dois números
separados). Sem swap (`SwapTotal: 0`), o gatilho de RAM do earlyoom fica sem proteção extra.
Fix aplicado: limpar /tmp de novo (lixo de OUTRAS ferramentas reacumula lá) + `systemctl mask
tmp.mount` (permanente após reboot, já feito). Quando o operador insistir que "não é RAM" com
número em mãos, vale a pena checar dois momentos diferentes no tempo antes de descartar a
hipótese, e consultar uma segunda opinião (Opus) em vez de só re-afirmar o diagnóstico anterior.

bug real achado só ao rodar benchmark de PP/TG: `--moe-cache-auto` decidia o split KV/experts
ANTES de olhar pra um `--num-tokens` explícito, então um pedido de 16576 tokens de KV virava
silenciosamente 8256 quando o layer do MTP deixava o orçamento mais apertado -> só estourava
como CUDA OOM confuso no meio de um prefill real, não no startup -> corrigido dobrando o
override no floor de `kv_reserve_tokens` (que já tinha um assert de recusa pronto, só faltava
alimentá-lo com o valor certo). Pedido explícito do usuário (`--num-tokens`) deve sempre virar
piso obrigatório, nunca sugestão descartável silenciosamente.

sintoma -> causa -> fix

MTP k=1 falha com `AssertionError: cache budget too small...` em 16K na RTX 5080 -> `--spec-mtp 1` aloca pesos adicionais (draft model) na VRAM, o que faz a margem padrão (`--mem-ratio 0.9`) ser violada por exatos ~126 MB, engatilhando o hard reject do scheduler -> usar `--mem-ratio 0.99` e, temporariamente para validação em 16GB, desativar o CUDA graph (`--no-graph`) para caber o KV exato. A cura real é plugar TurboKV no QSA.

MTP k=1 crasha com `ValueError: --spec-mtp > 0 requires overlap scheduling disabled` mas a env var estava no terminal -> se chamar via `bench_pp_tg.py`, a flag de ambiente precisa ser exportada ou passada imediatamente antes do binário python para que o subprocesso do `ft serve` herde o ambiente correto: `FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1 .venv/bin/python benchmarks/bench_pp_tg.py ...`


sintoma -> causa -> fix
MTP k=1 OOM (CUDA out of memory) em `chunk_gated_delta_rule_fwd` ao forçar `--mem-ratio 0.99` na RTX 5080 -> O ratio de 99% eliminou o `reserve_bytes` de segurança, permitindo ao `moe-cache-auto` alocar 2.96 GiB em slots de especialistas. O prefill dinâmico do GDN (que cria tensores de ativação em chunks de 8192) estourou os últimos megabytes físicos da placa. Conclusão: É fisicamente impossível sustentar 16K + MTP k=1 + MoE Cache viável em BF16 na placa de 16GB. O OOM não é bug de vazamento, é limite físico. Ação corretiva mandatória: integrar TurboKV (4-bit) ao QSA imediatamente para reduzir a pressão base de KV.

Phase 2 (TurboKV para QSA) concluída com sucesso:
O QSA requeria o uso do cache KV em BF16, pois os backends de turbo só tinham layout `(tokens, heads, groups)` denso, e o QSA opera com page tables. Descobrimos que o "token index" de um token em uma página alocada é simplesmente `slots = physical_page * PAGE_SIZE + page_offset`. 
Modificamos `qsa_pool.py` para usar composição dinâmica com `TurboMHAKVCache` quando `--kv-format turbo4` for passado, delegando `k_cache`/`v_cache` e reconstruindo os blocos de índice. Atualizamos `qsa_sparse.py` e o kernel do triton `qsa/attend.py` para passar os dicionários de turbo (`kn_ptr`, `vn_ptr`, `cent_ptr`, etc.) se `compressed=True`, chamando as rotinas `turbo_k_tile` e `turbo_v_tile`.
Economia confirmada: 16K requer apenas 60 MB de KV (em vez de 430 MB). 256K precisará de apenas 960 MB, tornando viável a RTX 5080 (16GB) para inferência com 256K de contexto sem sofrer OOM!

### Phase 2 Extension: TurboKV + QSA Compilation (PTX OOM)
- **Erro**: Tentar integrar o `turbo_k_tile` (descompressão de 4 bits em runtime) dentro do kernel denso `qsa_sparse_paged_attention` hibridizado resultou em um hang e crash por parte do host (`earlyoom`, `exitcode=-15`). O compilador JIT `ptxas` da NVIDIA não consegue otimizar as estruturas unrolled (decodificação de códigos via ponteiros cent/norm) junto com pipelining (`num_stages=2`), explodindo o uso de RAM no host para incríveis **70 GiB (VmRSS)**.
- **Acerto (Evitando o Erro)**: MTP=1 em 16K sem compressão gera `CUDA OutOfMemory` transiente. Tentar compressão TurboKV + QSA em 1 único kernel falha no host RAM. O correto é usar `MTP=0` com `--mem-ratio 0.96` para garantir ~28 tok/s em TG. Para contextos de 128K+ no futuro, a arquitetura deve DECOUPAR a descompressão (Kernel 1 = Descompressão em VRAM cache rápido FP16) do Sparse Attention (Kernel 2 = QSA clássico FP16), fugindo do estrangulamento do `ptxas`.

### Operacional e Monitoramento (Hygiene)
- **Erro**: Deixar múltiplos processos assíncronos de monitoramento (como `tail -f`) acumulando no background após o fim de benchmarks, além de manter subagents rodando em idle sem uso, o que degrada o ambiente e consome memória inutilmente (ex: tarefas "Wait for task X" acumuladas).
- **Fix**: Regra mandatória de higiene: sempre matar monitores e shells descartáveis (usar `killall tail` ou `manage_task kill`) assim que a análise terminar. Se um subagent não tem mais função imediata ou o escopo da pipeline for concluído, ele deve ser finalizado via `manage_subagents kill`. Jamais manter processos zumbis/inativos poluindo o log ou o host.

### Pre-Flight Checklist (Garantia de Ambiente Limpo)
- **Erro**: Iniciar novos benchmarks pesados de PP/TG sem conferir o estado prévio da máquina e placa de vídeo. Isso causa contaminação de resultados (throughput abaixo do real devido a swaps ou competição de processos) ou falhas catastróficas (OOM no kernel por resíduos de tarefas zumbis de VRAM ou RAM host, como as criadas por crashes anteriores ou earlyoom).
- **Fix (Regra Mandatória)**: Antes de QUALQUER novo teste ou benchmark neste projeto, deve-se realizar o "Pre-Flight Check":
  1. Verificar RAM livre do host via `free -h` (garantir cache limpo e nada em swap/tmpfs sem necessidade).
  2. Verificar VRAM da GPU via `nvidia-smi` ou utilitário equivalente (garantir que não há processos remanescentes comendo memória de vídeo ou cycles de compute).
  3. Verificar processos órfãos (`ps aux | grep -i python` / `killall tail`) para erradicar agentes zumbis antes do run.
  Só prosseguir com a execução após confirmação visual de que o hardware está 100% livre.
