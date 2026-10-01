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

## 8. Resultado da bissecção por grupo (atualizar)

(pendente: W4 = base W1 + `linear_state_pool`/`graph`/`scheduler`/`adaptive_mtp` revertidos; W5 = base W1 + norma/linear/attend/
mha_pool/vram_profile/engine/config/kernel attention revertidos.)
