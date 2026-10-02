# FreeToken-next — estado de fechamento documental (2026-10-01)

## Baseline congelada

- HEAD de referência: 4b27df2 (next), com kernels CPU Q2_K/Q3_K commitados.
- ISTA 16K: RAW k0 64.20 TG mínimo 64.05; MTP5 107.51 TG mínimo 107.28; três repetições por braço; SHA 76a5508fd576.
- Pfeiffer: MTP funcional, 90.67 TG+ na evidência histórica; o gargalo é serviço de experts CPU.
- Qwen3.6-35B UD NVFP4: MTP5 diagnóstico 173.00 TG médio, mínimo 169.95; matriz final ainda não congelada.

## Fechado

- Ladder KV automático e VRAM adaptativa/fingerprinted com histerese, geometria validada e fallback seguro.
- KV host FP8 nos caminhos suportados; combinações incompletas falham fechado.
- QSA aceito através da fronteira comprimida 127→128 passou no teste CUDA.
- Controlador MTP usa custo por token comprometido, não acceptance isolada.
- Kernels CPU Q2_K/Q3_K adicionados com testes de layout/dequantização.
- HybridRadix inseguro para qwen3_5_moe isolado; MTP sem prova de estado permanece fail-closed.
- Contrato: sequência final e estado comprometido equivalem ao RAW; igualdade bit a bit interna é apenas diagnóstico.
- Desempenho: maximizar TG, aceitar só a menor perda de PP necessária e rejeitar respostas lixo.

## Em aberto

1. AD IQ2_S/IQ4_NL: MTP k1–k5 ainda diverge no protocolo 16K. A primeira decisão errada ocorre no verify batched, antes do commit, em uma linha de baixa margem. O caminho parcial também falhou SHA (k0 a59f7ed31d3a, k1 7b8b387b07ba, decode32).
2. AD/NVFP4: uma correção de raiz e uma matriz curta padronizada; não abrir nova busca de micro-levers.
3. 35B: revalidar GGUF/NVFP4 e otimizar além de 190 TG sem alterar Qwen3.8.
4. Revisão final 16K: exatamente 3×RAW + 3×MTP, TG, PP cold/hot, SHA e qualidade.
5. Long context somente com autorização: Flash/MoE em 262144; dense em 64–128K; pressão, recuperação, needle e 20/20.
6. Separar commits aceitos das alterações/deleções de usuário antes do freeze.

## Veredito

NOT READY FOR LINUX PRODUCTION-HARDENING até fechar AD/NVFP4 MTP, matriz 16K, certificação longa e freeze reprodutível.

## Próximo caminho curto

1. Corrigir a causa AD no nível de decisão/estado comprometido.
2. Medir AD e NVFP4 em protocolo pareado, três repetições por braço.
3. Congelar apenas ganhos comprovados e executar os gates finais autorizados.

## Block 1 AD — fechamento 2026-10-01

- AD RAW 16K comprovado: 51,84–52,58 TG; o alvo >68 não é um baseline RAW demonstrado.
- AD MTP k5: 68,15 TG em uma matriz com SHA divergente; fail-closed por divergência pré-commit no verify batched.
- k2–k4 permanecem sem matriz pareada independente; não certificados.
- Ver [`docs/dev/BLOCK1-AD-20261001.md`](BLOCK1-AD-20261001.md).

## Block 1 AD — sessão 2 (2026-10-01)

- Verify em lote == RAW (SHA) em AD k1–k5; AD melhor k2 = 60,3 TG (RAW 51,4); k6 capado. Causas e regras: [`BLOCK1-SESSION2-20261001.md`](BLOCK1-SESSION2-20261001.md).
- Aberto: ISTA k5 108,4 (tag) -> ~98 TG nesta árvore (bissecção por arquivo em andamento); seleção automática de k; OOM de reserva; cache de prefixo híbrido.


## Block 1 AD — sessão 3 (2026-10-01)

- RAW AD provado: 52,7–52,9 TG, cert == HEAD (4 boots pareados). ISTA k5 104,7 -> 106,3 (cert 108,3; SHA == RAW); resta ~2 TG de journal QSA.
- Auto-k (sessão 3b, `bb02294`): AD escolhe k2 já na 1ª requisição (62,3–62,7, SHA == RAW); ISTA k5 107,2 em regime (era 96–98); empate resolve para o mais fundo.
- Paridade AD: prompts 0 e 7 == RAW em k1–k5; **prompt 3 diverge (pré-existente, `1bf00f3` igual)** — o desvio nasce no verify, camada 7 (QSA ou MoE/PLE), numa janela que fecha grupo comprimido; replay linha a linha corrige mas custa 47% do TG. Paridade do ISTA provada só no prompt 0.
- VRAM: reserva de prefill já é emprestada aos experts no decode; abertos: folga de 0,04 GiB no decode e OOM no ciclo especulativo derruba a requisição.
- Detalhe: `BLOCK1-SESSION2-20261001.md` seções 9 e 12. Veredito: NOT READY FOR BLOCK 2.
