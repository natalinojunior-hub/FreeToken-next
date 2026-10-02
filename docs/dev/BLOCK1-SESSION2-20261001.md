# Block 1 AD — sessão 2 (2026-10-01): causas raiz, correções, becos sem saída e regras

Escopo: AD IQ2_S/IQ4_NL (`Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64`), 16K, MTP k1–k5. Comparação cruzada com ISTA
(`Qwen3.8-Flash-Next-ISTA-IQ3_XXS/IQ3_XXS`). Commits: `26a4344`, `e860b40` e os seguintes desta sessão.
Este documento é o registro de **por que** cada coisa foi (ou não) alterada. Leia antes de mexer nas áreas listadas.

## 1. Veredito curto

| Item | Estado |
|---|---|
| Verify em lote == RAW (SHA) em k1–k5, graph on, 256 tokens, 3 repetições | **Sim** (SHA `235a97ef64b9`) |
| TG AD (RAW 51,4), 3 repetições, SHA igual ao RAW: k1 60,1 · **k2 61,9** · k3 58,1 · k4 55,0 · k5 52,2; auto 62,1–62,2 (1ª requisição 55,5 com calibração) | k2 é o melhor (+20%) |
| k6 | **Diverge** (caractere 18) e fica lento → capado em 5 |
| ISTA k5 (RAW 64,1) | **104,4 TG** (era 98,5 antes da correção 6 abaixo); tag certificada e `4b27df2` dão 108,4; falta ~4 TG = custo da transação QSA |
| Veredito geral | **NOT READY FOR BLOCK 2** (regressão ISTA, seleção automática de k, OOM de reserva) |

## 2. Causas raiz encontradas (todas confirmadas por experimento)

1. **`small_batch_linear` despacha 2–8 linhas por kernels diferentes do M=1** (split-K ou GEMM cuBLAS). Ordem de redução
   difere do decode → tokens/estados diferentes em linhas de baixa margem. Correção: `FREETOKEN_ROW_INVARIANT_LINEAR=1`
   (laço `F.linear` por linha, bit a bit igual ao M=1). Ligado por padrão **só** para AD IQ2_S/IQ4_NL (`engine/config.py`).
   - `torch.bmm` NÃO reproduz o `F.linear` de uma linha de forma confiável (falha em vários formatos). Não tente.
2. **Perfil do kernel QSA depende de `rows*kv_heads`** (`kernel/triton/qsa/attend.py`): parâmetro `row_invariant`
   fixa o perfil como se fosse 1 linha. Teste com controle negativo em `tests/models/qwen4_exp/test_qsa_kernels.py`.
   Sozinho não resolve o SHA; é necessário, não suficiente.
3. **Gated RMSNorm do GDN**: `calc_rows_per_block(M)` dava 1, 2 ou 4 linhas por bloco conforme M, mudando a redução
   (3–9% dos casos aleatórios diferiam). Fixado em 1 para M ≤ 1024 (`kernel/fla/layernorm_gated.py`). O decode RAW já
   usava 1 → **RAW inalterado**. Teste: `tests/kernels/test_gated_rmsnorm_row_invariant.py` (falha sem a correção).
4. **Rollback QSA completo no commit sem replay** (`scheduler/spec.py`): o primeiro `_restore_qsa_state(req, pre_draft=True)`
   fazia rollback total da transação e **zerava a linha comprimida do grupo fechado que foi aceito**; o rollback parcial
   (`keep_end`) só rodava em `finish_state`. Correção: pular o rollback total quando `zero_replay`.
   Sintoma: ciclos com grupo fechando (posição 4m+3) dentro da janela mantida + rejeição → logits do próximo passo
   divergem (até 8,1). Prova: `FREETOKEN_VERIFY_ORACLE_COMMIT=1` (próximo passo funcional), 24/24 ciclos limpos depois.
6. **`spec_state_steps` 2k+2 custava 10% do TG do ISTA (108 → 98) sem benefício**: bissecção por commit, depois por arquivo (árvores mistas W1..W7) isolou `kvcache/linear_state_pool.py`. A janela do verify tem exatamente k+1 linhas quando o commit não adia linhas (só o k1 adia, `SPEC_DEFER_MAX`). Voltou a `k+1` (+`SPEC_DEFER_MAX` se k==1). ISTA 98,5 → 104,4; AD +1,6 a +2 TG. Nunca superdimensionar buffers do verify "por segurança": comprova-se com matriz SHA, não com folga.
5. **A transação QSA obrigava replay em todo modelo QSA** (`qsa_requires_replay`), derrubando o ISTA para 30–41 TG
   (abaixo do RAW). `FREETOKEN_ENABLE_PARTIAL_SPEC` agora vale `1` por padrão (`_partial_spec_enabled()` em `spec.py`);
   `=0` restaura o replay.

## 3. Artefatos de medição que enganaram (não repetir)

- **MTP estava desligado nos "PASS" iniciais.** `config._safe_spec_mtp_depth` fechava AD IQ2_S/IQ4_NL (`spec_mtp=0`):
  os dois braços eram RAW. Sempre confira no log do servidor que há `spec: k=` e `[mtp-economics]`.
- **`FREETOKEN_VERIFY_ROWWISE_LINEAR` NÃO existe no código** (só em docs antigos). A "recuperação de paridade com rowwise+
  sequencial" citada nos relatórios anteriores era só `FREETOKEN_GDN_VERIFY_SEQUENTIAL`.
- **`FREETOKEN_GDN_VERIFY_SEQUENTIAL` com `keep_rows`** reutiliza `spec_states[li]` em cada chamada de 1 linha: linhas ≥1 não
  são gravadas. Qualquer SHA com essa flag é inválido para o estado comprometido. Não use como oráculo de commit.
- **Acerto no cache de prefixo muda a saída do próprio RAW**: k0 com prefixo em cache (`c1edcf1c1d66`) ≠ k0 com prefill novo
  (`6db8d9b3f514`). Em varreduras de um boot, **aqueça o k0 antes** (`"warmups": 1`) para todos os braços terem a mesma
  condição. É um defeito separado (HybridRadix) — registrado, não corrigido.
- **PP de k≥1 numa varredura é acerto de prefixo** (~14,5K tok/s) — ignore.
- **`bench-mtp-equiv.sh`** usa 1 repetição, 0 warmup e desliga overlap: não compare TG dele com a matriz de 3 repetições.
- **Árvore de trabalho (git worktree) não tem os `.so`**: copie `python/**/*.so` da árvore principal. Sem isso o servidor
  falha com `_pinned_tensor` e o `grep` do bench mostra só `Traceback`.
- **Trocar variável com servidor ligado não afeta o que foi capturado em CUDA Graph.** `FREETOKEN_ROW_INVARIANT_LINEAR`,
  `ROW_INVARIANT_QSA/NORM` são lidos em tempo de captura: definem-se **no boot**. `FREETOKEN_MTP_FORCE_DEPTH`,
  `FREETOKEN_ENABLE_PARTIAL_SPEC`, `FREETOKEN_DISABLE_QSA_TXN` são lidas em Python a cada ciclo e podem trocar em runtime.
- Um evento de monitor de **outra tarefa** (id que você não lançou) não é evidência: confira o log real.
- **OOM reprodutível**: o crescimento de residência de decode deixa reserva de 0,01 GiB e uma alocação de 108 MiB falha
  (modo oracle de commit). Contorno só de diagnóstico: `FREETOKEN_DECODE_RESIDENCY=0`. É risco de produção ("no hidden OOM").

## 4. Por que k3–k5 não superam k2 no AD (fatos medidos)

| | AD | ISTA |
|---|---|---|
| passo RAW | 19,4 ms (51,4 TG) | 15,6 ms (64,1 TG) |
| ciclo k5 / passo RAW | 3,5× | 3,5× |
| verify de 6 linhas / passo RAW | 3,0× | 3,0× |
| tokens por ciclo k5 | 3,3 | 5,3 |
| aceitação do 1º..5º rascunho | 78/57/40/33/24% | 100/94/88/73/73% |

Custo do motor por linha é igual em proporção; a diferença é a **aceitação**. No AD 66% das rejeições têm alvo pouco
confiante (margem < 2), no ISTA 100% têm margem ≥ 2 → o alvo IQ2_S é "mais ruidoso". **Não** se sabe ainda se é a cabeça,
o caminho do rascunho ou a quantização do alvo; o teste que separa é medir a cabeça com estado do alvo sempre correto.
Estágios do ciclo (ms, AD): rascunho 4,7–8,0 · verify 24,2 (2 linhas) / 35,7 (k2) / 43,0 (k3) / 58,1 (k5) · commit ~1.

## 5. Ferramentas criadas (use em vez de reiniciar servidor)

- `scripts/bench-sweep.py` + `FREETOKEN_RUNTIME_ENV_FILE`: **um boot, várias configurações**. JSON `[{name, env, warmups, repeats}]`;
  o primeiro item é a referência de SHA. Imprime `MATCH`/`DIVERGE@charN`. Reduz ~60–80% do tempo (boot=58 s, teste=~10 s).
- `scripts/mtp_verify_oracle.py` (servidor sob `python -m scripts.mtp_verify_oracle`):
  - `FREETOKEN_VERIFY_ORACLE_CYCLE=1,3,6|all` · `..._LIGHT=1` (só argmax e margem por ciclo) ·
    `..._COMMIT=1` (estado pós-commit vs RAW + **logits do próximo passo** do estado real vs refeito + primeira camada QSA) ·
    `..._QSA_LAYER`, `..._MODULE_LAYERS` (captura por submódulo; primeira diferença) · `..._ROWWISE_ALL` (diagnóstico).
  - O oracle compara verify × RAW **a partir do mesmo estado**: não detecta estado já corrompido; por isso o modo COMMIT.
- `FREETOKEN_DEBUG_SPEC_TIMING=1` (estágios) · `FREETOKEN_DEBUG_QSA_TXN_TIMING=1` (tempo de relógio por método da transação).
- `py-spy record -r 250 -f raw --subprocesses -- python scripts/bench-sweep.py ...` e agregar só quadros dentro de `run_spec_step`.
- Perfil: ~88–90% do ciclo é `synchronize` (espera de GPU). Perda de TG = GPU parada em bolhas por trabalho de CPU no caminho crítico.

## 6. Em aberto (não dar como resolvido)

- **Regressão ISTA k5: 108,4 (tag `ista-16k-certified-20260930` e `4b27df2`) → 96,4 (`e860b40`, caminho sem replay) → 98,7 (árvore atual).**
  Bissecção: desligar a transação QSA dá +3 TG (101,3); `spec.py` e `qsa_sparse.py` do commit rápido sobre a árvore atual
  (W1) dão 98,7; as invariâncias (norma/QSA) são neutras ou levemente positivas. **O resto está em outros arquivos
  alterados** (`linear_state_pool` 2k+2, `graph.py`, `scheduler.py`, `adaptive_mtp.py`, `mha_pool`, `engine.py`, ...). Resultado
  das árvores W4/W5 (grupos revertidos): ver seção 8 (atualizada ao fechar).
- Seleção automática de profundidade: calibração com 4 amostras por profundidade e permanência no k máximo pedido salvo
  vitória clara; dá 54,7/60,3/54,7 no AD (k2 = 60,3). Perfil salvo pode ficar velho (foi salvo `depth=5` após varreduras forçadas).
- k6 diverge (char 18) — capado em 5 (`_AD_MAX_PROVEN_MTP_DEPTH`).
- OOM de reserva de residência (seção 3). Defeito do cache de prefixo híbrido (seção 3).
- Verify linha a linha é um laço de kernels; um GEMV único com ordem de redução fixa por linha reduziria custo (cada linha de verify ≈ 8–10 ms no AD).

## 7. O que mexer em cada coisa causa (mapa de efeitos)

| Mexer em | Efeito | Cuidado |
|---|---|---|
| `small_batch_linear` (linear sem quantização, 2–8 linhas) | muda ordem de redução do verify | só AD liga o laço; outros modelos usam split-K/GEMM (mais rápidos, não row-exatos) |
| `calc_rows_per_block` | muda a redução da norma com gate | RAW usa M=HV (já 1 linha/bloco); aumentar o limite quebra paridade |
| `attend.py` perfil por linhas | muda splits/tiles do QSA | `row_invariant` só quando `spec_logits_indices` existe |
| `spec.py` rollback QSA | apaga/preserva escritas aceitas | nunca rollback total quando `zero_replay` |
| `FREETOKEN_ENABLE_PARTIAL_SPEC=0` | volta ao replay (exato, lento: AD 25–30, ISTA 30–41 TG) | só diagnóstico |
| `_AD_MAX_PROVEN_MTP_DEPTH` | k>5 não provado | não subir sem matriz SHA |
| `spec_state_steps` (2k+2) | buffers do verify cobrem janela com prefixo diferido | tests de orçamento de bytes dependem disso |

## 8. Resultado da bissecção por grupo

W4 (base W1 + `linear_state_pool`/`graph`/`scheduler`/`adaptive_mtp` revertidos) = 108,0; W5 (outros arquivos revertidos) = 99,9;
W6 (só `linear_state_pool`) = 108,2; W7 (`scheduler`/`adaptive_mtp`/`graph`) = 98,8. **Único culpado: `spec_state_steps` 2k+2.**
Árvores mistas: copiar `python/` da árvore atual e sobrescrever arquivos com `git show <commit>:path`; PYTHONPATH aponta para a cópia.

## 9. Cabeça MTP e limites de profundidade (medidos)

- A cabeça do AD (`...-AD-...-mtp.gguf`, sha256 `f521868a9e143718…`) é **byte a byte idêntica** à do ISTA e à do pfeifferj
  (`cmp`). Logo a cabeça não explica a aceitação menor do AD (76–78% no 1º rascunho contra ~100%).
- Cabeça candidata `unsloth/Qwen3.8-Flash-Next-GGUF` `MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (sha256 `cd87e5d1a4dadaee…`, 3,9 GB),
  em `/models/heads/unsloth/MTP/`: mesmos 32 tensores e formas da atual, 17 com tipo de quantização de maior precisão, mais
  `output.weight` e `token_embd.weight`. Uso: `FREETOKEN_MTP_PATH=<arquivo>`. No AD: k2 **63,0–63,3** (atual 60,1), k3 59,6 (54–59),
  k5 52,8 (50,3); aceitação do 1º rascunho 83% (76%); **SHA igual ao RAW** em todas as rodadas. Ganho de ~5% no melhor k. Não é o
  padrão (download externo de 3,9 GB).
- A aceitação baixa do AD **ainda não é provada como limitação do modelo**: falta uma referência independente (o `llama-server` em
  `/models/servers/llama-turbo-optimal/build/bin` valida k=1 contra o upstream; k≥2 não é certificado lá).
- ISTA com profundidade maior: k5 103,8 · k6 91,2 · k7 89,3 (SHA = RAW). Cada linha extra custa 8–10 ms e a 6ª posição só rende ~0,7 token → **k5 é o ótimo**.
- Custo por linha do verify (decode puro, razão k5/k0 por chamada de camada): ISTA mixer ×1,81 · MoE ×2,60; AD mixer ×1,72 · MoE ×2,68.
  Os dois modelos escalam igual: o AD não tem defeito próprio no motor. ~2/3 do custo extra é MoE, ~1/3 é mixer (GDN `_conv_decode`
  por linha + recorrência + QSA). Alvos: fundir o laço de convolução por linha do GDN; execução dos experts por linha.
- Mapa de oportunidades fora da recuperação dos 108 TG: cabeça Q8_0 (+5% AD) · calibração do k automático (1ª requisição −12%) ·
  PP frio dos DOIS modelos (~1,3–1,6K tok/s: 2 blocos de 8192 com 0 em cache, ~6–7 s por bloco; hipótese não testada: o prefill transmite os experts por bloco, então bloco maior reduziria passagens) · fusão da convolução por linha do GDN.

**Erro corrigido:** uma versão anterior afirmou que o PP frio do AD era 10× pior que o do ISTA. Era comparação de requisição fria com requisição em cache de prefixo (a varredura aquece o k0). O PP frio é igual nos dois.

## 10. Travamento silencioso de boot (causa raiz, corrigida)

- **Sintoma:** servidor parado após `Free memory before loading model`, GPU 0%, CPU ociosa, log mudo por minutos; 37 de 43 threads em `futex`,
  uma thread em `nanosleep`; ~1 MB lido do disco. Apareceu em 3 execuções seguidas, nas árvores certificada e atual.
- **Causa:** a pilha (faulthandler) mostrou `weight.py → qwen4_exp/gguf.py:_to_bf16 → kernel/gguf.py → torch.utils.cpp_extension._jit_compile →
  torch/utils/file_baton.py:wait`. Um job meu foi interrompido no meio do build JIT de `freetoken_gguf_kernels` e deixou
  `~/.cache/torch_extensions/py312_cu130/freetoken_gguf_kernels/lock`; o `FileBaton` do torch espera essa trava **sem limite**.
- **Gatilho:** criar `git worktree` novo muda o caminho das fontes (`build.ninja`) e força recompilar a extensão; interromper nesse instante deixa a trava.
- **Correção no motor:** `kernel/gguf.py::_clear_stale_build_lock` remove a trava se tiver > 60 s e não houver compilador rodando
  (ninja/nvcc/cicc/ptxas/cc1plus...). Teste: `tests/kernels/test_gguf_jit_lock.py`. Verificado: trava plantada com 10 min foi removida no boot (log `removed stale JIT build lock`).
- **Correção nas ferramentas:** `benchmarks/bench_pp_tg.wait_ready` detecta boot mudo (`FREETOKEN_BENCH_BOOT_STALL_S`, padrão 180 s), envia SIGABRT
  ao grupo de processos e despeja as pilhas (servidor sobe com `PYTHONFAULTHANDLER=1`), e falha com `BOOT STALL`.
- **Regras:** nunca interromper (SIGINT/SIGKILL) um boot durante o primeiro build JIT; para A/B entre árvores de trabalho use `TORCH_EXTENSIONS_DIR`
  próprio por árvore; mate processos por PID (`pkill -f` com o nome do script mata o próprio shell); para parar um bench use SIGINT
  (o `finally` encerra o servidor) e confirme com `tail --pid`, sem laço de espera.
- **Armadilha de medição:** sem `--spec-mtp` o padrão do servidor é 6 (limitado a 5 no AD com o gate aberto), então "RAW" medido sem a flag
  não é RAW puro. Para RAW use sempre `--serve-arg="--spec-mtp 0"`.

## 11. Estado final desta sessão e próximos passos

- Commits desde `4b27df2`: 26a4344, e860b40, 056f33d, 422692a (buffers k+1), 95e14f8 (docs), f21780b (docs PP), 7024599 (trava JIT), 0376095 (formato), 88c3f65 (prompt).
  Os 6 últimos foram refeitos sem o `docs/dev/campaign36-mtp-audit.zip` (680 MB) que entrou por engano em `git add docs/dev`; hashes antigos (4cf1c27, 363303f, 101b021, 39bc5d5, 2b97d3b, aeaedcb) só existem em `refs/backup/pre-zip-fix`.
  Regra: nunca `git add <diretório>` com a árvore suja; listar arquivos explícitos e conferir `git diff --cached --stat`. CI: formato e lint limpos, mypy 447 arquivos, **2768 testes**.
- Medido: AD k1–k5 == RAW (SHA 235a97ef64b9), k2 61,9 TG (cabeça Q8_0 opcional 63,1; RAW 52,9); ISTA k5 104,4 (tag 108,4).
- **Abertos (não dar como resolvidos):** (1) RAW do AD: gate sem prova, execução atual instável 43–53; (2) ISTA −4 TG; (3) calibração do k automático.
- Veredito: **NOT READY FOR BLOCK 2.** Prompt da próxima sessão: [`PROMPT-NEXT-SESSION-block1.md`](PROMPT-NEXT-SESSION-block1.md).
- Lição de método: o `--spec-mtp` padrão é 6 e liga o MTP com o gate aberto; "RAW" medido sem `--spec-mtp 0` não é RAW.

## 9. Sessão 3 (2026-10-01): fechamento dos 3 pontos abertos

Método: boots intercalados, árvore certificada (`ista-16k-certified-20260930`, worktree própria, `.so` copiados, `TORCH_EXTENSIONS_DIR` por árvore) contra HEAD, mesma hora, `--spec-mtp` explícito, 16K, 256 tokens. Dados: `final-closure-20261001/results/session3/`.

1. **RAW do AD: provado.** `--spec-mtp 0`, 5 repetições por boot, 2 boots por árvore: cert 52,85 / 52,76; HEAD 52,87 / 52,73 (mín. 52,67). Sem oscilação 43–53 em nenhum boot. A oscilação 43–53 **não se reproduziu** em 4 boots limpos; causa não provada (hipóteses: jobs concorrentes na máquina; `Decode residency` alternando 3724↔4716 slots a cada requisição nos logs antigos). Se voltar, rode boot isolado com telemetria. Gate cumprido: RAW 52,7–52,9 estável, HEAD == cert.
2. **ISTA k5: 104,7 → 106,3 TG (SHA `76a5508fd576` == RAW).** Cert 108,38 / 108,28; HEAD antes 104,67 / 105,76. Decomposição por flag (HEAD): `ROW_INVARIANT_QSA=0` +1,1 TG; `ROW_INVARIANT_NORM=0` 0; `DISABLE_QSA_TXN=1` +1,8 TG.
   - Correção A: `FREETOKEN_ROW_INVARIANT_QSA` agora só liga por padrão com `FREETOKEN_ROW_INVARIANT_LINEAR=1` (AD). O perfil do QSA com linhas invariantes só foi necessário para o AD; ISTA mantém SHA == RAW sem ele.
   - Correção B: `prepare_spec_txn` fazia 4 sincronizações com o host (`unique`/máscaras) por ciclo; agora 1 (`tolist` único e conjuntos montados no host). Tempo da transação por ciclo: prepare 0,65 → 0,35 ms, snapshot 1,0 → 0,58 ms.
   - Resta ~2 TG (106,3 contra 108,3): custo residual da transação QSA (journal). **Alvo ≥107 não atingido.** Próximo passo: passar as posições de escrita já conhecidas no host (lista do scheduler) e eliminar a última sincronização.
3. **Seleção automática de k: sem mudança de código.** Perfil limpo (`FREETOKEN_MTP_PROFILE=off`), `warmups 0`, 2 boots: 1ª requisição 58,6 / 58,6 TG, seguintes 59,7 (SHA == RAW). O número antigo "55,5 contra 62,0" não se reproduziu; a lacuna real é 1,9% na 1ª requisição e 3,7% contra o melhor k fixo (k2 = 61,9). Causa: a medição interna do controlador penaliza k2 (16,0 ms/token contra 14,7 do k1) enquanto a matriz externa mostra k2 3% melhor; é viés sistemático da sondagem intercalada, não ruído. **Testado e descartado:** `_PROBE_REPEATS` 4 → 8 (mesma escolha k1, 1ª requisição 58,2; sem ganho). Alvo: medir custo do verify por linha em vez de por profundidade intercalada (estimar TG(k) a partir da aceitação por posição + custo linear por linha). Não implementado.

Veredito da sessão 3: **NOT READY FOR BLOCK 2** (ISTA 106,3 < 107, seleção automática não otimizada, OOM de reserva e cache de prefixo híbrido abertos).

## 12. Sessão 3b (2026-10-02): seleção de k, custo do motor, paridade fora do prompt 0

Commits: `bb02294` (journal QSA em lote + k automático por aceitação agregada + sem cópia redundante no aceite total),
`543f9e1` (planejador não planeja chunk acima de `--max-prefill-length`), `0ddf904` (`--prompt-offset` fora do corpus falha em vez de repetir o prompt 0).
Dados brutos: `/models/desenvolvimento/tmp/rawab/` (`j_*`, `f_*`, `t_*`, `o*`, `h*`, `c*`, `k2_*`, `sweep_*`, `oracle_commit_p3.out`).

**Ponto 2 (ISTA) — causa da queda 108 → 96–98 com k automático:** o controlador (regra "empate fica raso") escolhia k3 no
ISTA porque a sondagem de aquecimento infla ciclos fundos (sobrecusto fixo ~11 ms/ciclo e poucas tentativas nas últimas
posições). Árvores mistas: só `spec.py` revertido 108,30; HEAD 96,46 (k3). Correção: empate resolve para o **mais fundo**
(`_deepest_clear_winner`, margem 0,97). Replay exato das sondagens gravadas: ISTA k5, AD k2. Resultado (k automático, prompt 0):
ISTA 107,2–107,3 em regime (cert 108,4); AD 62,3–62,7 já na 1ª requisição, SHA == RAW.
Custo do motor com k5 fixo: cert 106,8/108,4 × HEAD 104,5/106,5 (≈ −1,9 TG, ruído entre boots ±1). Sem a cópia redundante
`commit_spec_row` no aceite total: +1,1/0 TG; AD k0–k5 18/18 SHA == RAW. Com k2 fixo (prompt 7): HEAD 82,9 / cert 83,1 / HEAD sem
transação 83,7 — a transação custa ~0,9% em k2.

**Teto 6 (padrão) × teto 5:** com k5 fixo no prompt 0 (único par com SHA igual) o teto 6 dá 97,8 contra 106,2 e, com k
automático, 92,6 contra 102,2 — os rascunhos mudam (aceitação 5,24 × 5,67 token/ciclo); `--kv-tiering off` com teto 5 reproduz
exatamente o mesmo padrão (mesma 1ª divergência na posição 103). Não é leitura de memória não inicializada (buffers com NaN:
idêntico). Nos prompts 3/7 a **saída do ISTA muda com teto/layout** (p3 dc9d ≠ 262c, p7 700e ≠ 5f16; cert p3 = dc9d), então
esses pares não são comparáveis e a paridade do ISTA só está provada no prompt 0 (comparação RAW × MTP entre boots também tem
layout diferente). **Decisão sobre o teto padrão: indeterminada, sensível ao layout; mantido em 6 até haver pares com SHA igual.**

**Paridade AD fora do prompt 0 (achado novo, pré-existente):** prompt 7 k1–k5 == RAW (k2 71,0 TG). Prompt 3: k1–k5 todos com o
mesmo SHA `c250ac983c40` ≠ RAW `fc45c4e520d0`, 1º token divergente no índice 25 (`' at'` × `':'`), linha 0 do verify logo após
um aceite total k1. Igual na árvore `1bf00f3` completa (anterior à sessão) → não é regressão desta sessão. Classificação:
`FREETOKEN_ENABLE_PARTIAL_SPEC=0` (commit com replay linha a linha) **== RAW** mas TG 27,5 × 51,9; `DISABLE_QSA_TXN=1` diverge igual;
laço por linha no linear GGUF não muda. Oráculo de commit (`scripts/mtp_verify_oracle.py`, `FREETOKEN_VERIFY_ORACLE_COMMIT=1`,
`FREETOKEN_DECODE_RESIDENCY=0`): 1 de 35 ciclos com estado ≠ RAW (ciclo 30, posições 16402→16404, janela que fecha um grupo
comprimido 16403 % 4 == 3); camadas lineares 0–5 iguais, 6+ diferentes, e só nesse ciclo o KV do pool difere em ~10× mais
elementos (2575 × ~256 nos ciclos bons) → o desvio nasce **dentro do verify, na camada 7** (atenção QSA comprimida ou seu
MoE/PLE com 2 linhas). O gravador QSA do modo commit só cobre o passo seguinte, então não separa QSA de MoE/PLE.
**Aberto:** localizar o módulo da camada 7 (oráculo por módulo no ciclo 30) e torná-lo invariante por linha, ou replay só nesse caso.
O gate de paridade passa a exigir ≥ 3 prompts (`--prompt-offset 0 3 7`).

**Seleção automática entre textos:** prompt 3 HEAD 87,1 × cert 84,3; prompt 7 HEAD 78,8/81,5/81,1 × cert 84,4 (k2 fixo HEAD
82,8–82,9 estável). Na cert, os ciclos k2 em regime ficam ~3% mais rápidos depois da sondagem dela (provável histórico do cache
LRU de experts); não confirmado sem estatística de acerto do cache.

**Controlador de VRAM (estudo):** prioridade do solver = contexto (KV) → maior chunk de prefill → experts. A reserva do prefill
(~2 GiB a chunk 8192) **já é emprestada aos experts durante o decode** (`Decode residency` 4761 → 5884 slots e volta antes de cada
prefill). Chunk 4096 (5218 slots) deu 102,9 × 106,6, mas com aceitação diferente (5,14 × 5,38 token/ciclo, mesmo SHA): o
efeito dos slots extras **não foi medido** de forma isolada. KV em RAM no ISTA 16K libera só ~110 MiB (~65 slots, 1,4%).
Defeitos achados: (1) o crescimento de residência deixa só 0,04 GiB livres — qualquer alocação extra no decode faz OOM;
(2) OOM dentro do ciclo especulativo derruba a requisição (`_spec_step_or_fail` não tem rollback + nova tentativa, o forward
normal tem); (3) a calibração de VRAM reaproveitada planejava chunk acima de `--max-prefill-length` (corrigido em `543f9e1`).

Veredito sessão 3b: **NOT READY FOR BLOCK 2** — paridade AD falha no prompt 3 (pré-existente), ISTA k automático 107,2 < 108,4,
OOM no ciclo especulativo e folga de 0,04 GiB abertos.
