# FreeToken-next — lições operacionais e de fechamento

## Correção MTP

- Igualdade bit a bit de todos os intermediários não é requisito de produto. O gate é sequência final, rollback/replay, ausência de vazamento especulativo e estado recorrente equivalente após commit.
- QSA, GDN, KV e transações precisam ser comparados depois da aceitação/rejeição e do replay. Um delta interno isolado é evidência de diagnóstico, não prova de corrupção.
- Verify batched pode mudar decisões em linhas de baixa margem mesmo quando os primeiros tokens coincidem. O teste deve procurar a primeira decisão futura divergente.
- Fallback sequencial amplo pode recuperar correção e ainda destruir TG; não é solução aceita sem medir o custo e provar estado completo.

## Desempenho

- TG é o objetivo primário. PP deve permanecer o mais próximo possível do baseline; queda só é aceitável quando o ganho real de TG a justifica.
- Três repetições por braço são o padrão: 3×RAW + 3×MTP. Comparações devem ser pareadas na mesma sessão.
- Saída curta, lixo ou incoerente não conta como TG/PP ganho.

## VRAM e contexto

- O controlador deve aprender geometria/pressão real e recuperar capacidade quando a pressão externa desaparece.
- VMM-backed e alocações do allocator precisam ser contabilizados separadamente; não inferir uso físico apenas por torch.cuda.memory_allocated().
- KV host e ladder FP8→Turbo4→Turbo3 devem permanecer específicos ao caminho que provou equivalência; não generalizar uma otimização da família Flash para 35B sem evidência.

## Encerramento

- Depois que os gates forem fechados, parar a busca aberta de micro-otimizações. Registrar NO-GO, PHYSICAL, UNSUPPORTED ou BLOCKED com evidência, congelar fonte e empacotar auditoria pequena.

## Verify em lote e MTP (sessão 2, 2026-10-01) — `sintoma -> causa -> fix`

- "PASS" em k1–k5 com TG idêntico ao RAW -> o gate `_safe_spec_mtp_depth` zerava o MTP, os dois braços eram RAW -> confira `spec: k=` e `[mtp-economics]` no log antes de aceitar qualquer PASS.
- Verify diverge do RAW em linhas de baixa margem -> `small_batch_linear` usa kernel diferente de M=1 para 2–8 linhas (e `torch.bmm` também não reproduz) -> laço `F.linear` por linha (`FREETOKEN_ROW_INVARIANT_LINEAR`), só AD.
- Diferença rara (3–9% dos casos) no mixer GDN com mesma entrada -> `calc_rows_per_block(M)` muda a redução da norma com gate -> fixar 1 linha/bloco para M ≤ 1024 (RAW já usava 1, inalterado).
- Primeiro desvio na primeira camada QSA com q/k/v iguais -> perfil do kernel de atenção QSA depende de `rows*kv_heads` -> parâmetro `row_invariant` (necessário, não suficiente).
- Commit sem replay deixa o próximo passo errado (logits até 8,1) com estado linear igual -> o rollback QSA total (`pre_draft=True`) rodava antes do parcial e zerava a linha comprimida do grupo aceito -> pular o rollback total quando `zero_replay`.
- ISTA MTP 30–41 TG (abaixo do RAW) -> transação QSA forçava replay em todo modelo QSA -> `FREETOKEN_ENABLE_PARTIAL_SPEC` padrão 1.
- k0 com cache de prefixo != k0 com prefill novo -> cache de prefixo híbrido muda a saída do RAW -> aquecer o k0 antes (`warmups`) e comparar todos os braços sob a mesma condição.
- `FREETOKEN_VERIFY_ROWWISE_LINEAR` "recupera paridade" -> a flag não existe no código; era só `GDN_VERIFY_SEQUENTIAL`, que ainda não grava as linhas ≥1 em `spec_states` -> não usar como oráculo.
- Comparação com número histórico (107,5) sem reproduzir -> medir a **árvore certificada na mesma hora** (git worktree + copiar `.so`); AD/ISTA com bissecção por commit, depois por arquivo.
- Mudar flag em runtime sem efeito -> valor baked na captura do CUDA Graph -> definir no boot; só flags lidas em Python por ciclo trocam em runtime (`bench-sweep.py`).
- Evidência vinda de monitor/tarefa que não lancei -> não é evidência; ler o log real. Parar cada monitor ao receber o resultado (`TaskStop` só aceita `task_id`; parâmetro extra cancela o lote inteiro).
- Aceitação MTP baixa no AD (24% no 5º rascunho contra 73% no ISTA) -> alvo IQ2_S pouco confiante (66% das rejeições com margem < 2) -> não é defeito do motor; não gastar tempo otimizando kernels para recuperar aceitação.
