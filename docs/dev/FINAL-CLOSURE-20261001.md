# FreeToken-next — fechamento documental 2026-10-01

## Escopo

Este relatório cobre o delta desde open-audit-20260929.zip e registra o estado atual do worktree, os resultados comprovados e o que ainda impede o Linux production-hardening. O pacote correspondente é final-closure-20261001/freetoken-next-audit-20261001.zip.

## O que foi fechado

- Baseline ISTA 16K congelada: RAW k0 64.20 TG mínimo 64.05; MTP5 107.51 TG mínimo 107.28; SHA 76a5508fd576.
- Controller MTP usa custo por token comprometido, histerese e perfis persistentes; não escolhe profundidade apenas por acceptance.
- VRAM/KV ladder e perfis adaptativos têm testes de geometria, pressão e recuperação; combinações sem suporte falham fechado.
- QSA aceito através da fronteira 127→128 passou no teste CUDA dedicado.
- Kernels CPU Q2_K/Q3_K foram adicionados no commit 4b27df2, com testes de layout e dequantização.
- Pfeiffer MTP funcional e acima de 90 TG na evidência histórica; o gargalo restante é miss/serviço de experts CPU.
- HybridRadix inseguro para qwen3_5_moe foi isolado; MTP 35B sem prova de estado permanece fail-closed.
- Contratos de correção e desempenho foram registrados no ai-memory e nesta documentação.

## Investigação AD/NVFP4

O caminho AD IQ2_S/IQ4_NL ainda não é certificável. O trace 16K mostra que a primeira decisão errada ocorre dentro do verify batched, antes de commit/replay, em uma posição de baixa margem. Desativar CUDA graphs não resolveu. O cálculo HC de referência recupera os tokens de um ciclo, mas uma execução completa ainda diverge e perde TG. O caminho parcial existente também falhou SHA: RAW a59f7ed31d3a, MTP 7b8b387b07ba, decode32. Portanto não há patch de produção aceito nesta continuação.

Não é necessário igualar cada logit; é necessário impedir que a diferença altere decisão futura ou estado comprometido. Uma solução sequencial ampla não atende ao objetivo de TG e não foi promovida.

## Resultados e limites

| Área | Estado | Evidência |
|---|---|---|
| ISTA 16K | WIN/certificado | tag ista-16k-certified-20260930, SHA estável |
| Pfeiffer 16K | WIN funcional/performance histórica | 90.67 TG+, miss CPU medido |
| AD MTP | BLOCKED/NOT CERTIFIED | divergência pré-commit no verify batched |
| Radix | RAW funcional; MTP/long-context pendente de matriz final | matriz e relatório anterior |
| 35B NVFP4 | funcional em RAW; MTP fail-closed até prova | b27a5ae e final-closure-20260930/35b-cert |
| VRAM controller | WIN em testes unitários/pressão; certificação final pendente | planner/profile tests e logs anteriores |
| 256K | não encerrado | requer autorização e matriz final |
| dense long context | limite físico; usar 64–128K conforme capacidade | regra do projeto |

## Classificação dos itens restantes

- **WIN:** ISTA 16K, Pfeiffer funcional, ladder KV/VRAM adaptativa, QSA 127→128 e kernels Q2_K/Q3_K.
- **NO-GO:** HybridRadix warm-prefix em qwen3_5_moe; fallback sequencial amplo para AD por custo de TG.
- **PHYSICAL:** dense acima de 128K nesta versão; miss/serviço CPU do Pfeiffer.
- **UNSUPPORTED:** MTP externo para famílias 35B sem head implementado; certificar RAW/vision nesses modelos.
- **BLOCKED:** AD/NVFP4 MTP, matriz final 16K, freeze reprodutível e certificação 262144 enquanto não autorizada.

## Alterações desde a auditoria

- 4b27df2: kernels CPU Q2_K/Q3_K.
- 86ef037: AD IQ2_S/IQ4_NL MTP fail-closed até paridade de tokens.
- b27a5ae: Qwen3.5 speculative decoding fail-closed sem estado provado.
- 4fa90fe: quarentena HybridRadix warm-prefix para Qwen3.5.
- 1716b74, 8c0a6ce, 9d0180c, 02972e2, 75824ae, 58138f4 e 4aed3da: proteção de host-tier, protocolo de repetição, raw explícito, atenção, VRAM e rollback.
- Worktree atual contém mudanças/deleções de usuário não relacionadas; nenhum freeze amplo foi feito para não sobrescrevê-las.

## Próximos passos mínimos

1. Corrigir AD/NVFP4 na causa de decisão/estado, com uma única nova direção de patch sustentada por evidência.
2. Repetir matriz 16K padronizada: 3×RAW + 3×MTP, TG, PP cold/hot, SHA e qualidade.
3. Separar e commitar apenas o runtime aceito; preservar alterações de usuário.
4. Com autorização, rodar Flash/MoE em 262144 e dense em 64–128K, incluindo pressão, recuperação, needle e 20/20 qualidade.
5. Executar make ci, Ruff e MyPy no source congelado e gerar o pacote final.

## Veredito

**NOT READY FOR LINUX PRODUCTION-HARDENING.** O ISTA e os ganhos comprovados permanecem preservados; AD/NVFP4 MTP, a matriz final e a certificação longa ainda não estão encerrados.

## Addendum — Block 1 AD (2026-10-01)

O relatório detalhado está em [`BLOCK1-AD-20261001.md`](BLOCK1-AD-20261001.md). A evidência corrigida mostra que o alvo histórico `>68 TG` pertenceu ao AD MTP diagnóstico; o AD RAW arquivado permanece em `51,84–52,58 TG`, sem regressão comprovada acima de 68. O AD MTP k5 ainda diverge dentro do verify batched antes do commit, e k2–k4 não têm artefatos pareados independentes suficientes para certificação. O estado permanece **NOT READY**.
