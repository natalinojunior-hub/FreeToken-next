# FreeToken-next — Block 1 AD: fechar os 3 pontos abertos (nova sessão)

MODO: ACT/AUTÔNOMO. Sonnet 5.5 executa; Opus 5.5 só como advisor em decisão de arquitetura/numérica/kernel. Serialize a GPU (`flock /tmp/gpu.lock`).
Tudo event-driven: sem sleep/polling/timeouts; job em segundo plano + Monitor; pare cada monitor com `TaskStop` (só `task_id`).
Não suba servidor sem hipótese e sem o vigia de boot ativo. Mate processo por PID (nunca `pkill -f <nome do script>`).

LEIA ANTES (nesta ordem): ai-memory (`_rules/mtp-verify-do-not-repeat.md`, `_rules/boot-hang-stale-jit-lock.md`,
`notes/block1-ad-parity-findings-20261001b.md`), `docs/dev/BLOCK1-SESSION2-20261001.md`, `docs/dev/LESSONS.md`. Código vivo vence documento.
Escopo: AD IQ2_S/IQ4_NL, 16K, e ISTA só para A/B. Não reabra NO-GOs (seção 3 e 10 do documento) sem mecanismo novo.

## Estado herdado (HEAD no `next`)
Verify em lote == RAW (SHA 235a97ef64b9) em AD k1–k5; AD k2 61,9 TG (RAW 52,9); ISTA k5 104,4 (tag certificada 108,4); CI limpo (2768 testes).

## Os 3 pontos
1. **RAW do AD não provado.** Árvores com gate fechado (cert, 4b27df2, 26a4344, e860b40) deram 52,7–52,9; a atual, `--spec-mtp 0`, oscilou
   43–53 TG dentro do mesmo boot e a causa não foi achada (RAM, temperatura e OOM aprendido foram descartados). Medir RAW **sempre com `--spec-mtp 0`**,
   mesma hora, árvores intercaladas, `TORCH_EXTENSIONS_DIR` próprio por worktree, telemetria (CPU/GPU/NVMe temp, clocks, `/proc/pressure`) por repetição.
   Gate: RAW dentro de ~51,8–52,9 estável, ou diferença explicada por A/B pareado. Se for regressão, bissecção por commit e por arquivo.
2. **ISTA 104,4 contra 108,4 TG** (−4 TG). Transação QSA custa 1–2 TG; o resto não tem causa. Bissecção por arquivo (árvores mistas) sobre a base atual.
   Diário em bloco para bf16 foi testado e **descartado** (neutro). Alvo: ≥107 TG com SHA = RAW, sem perder a correção do limite 127→128.
3. **Seleção automática de k.** 1ª requisição no AD roda 55,5 TG contra 62,0 das seguintes (calibração de 4 amostras por profundidade; perfil salvo pode ficar velho).
   Escolher por TG de tokens comprometidos, com histerese e fallback k0; convergir ao melhor k (AD k2) já na 1ª requisição curta.
   Valide com matriz 3 repetições, SHA = RAW, e perfil limpo.

## Fim (obrigatório)
Testes focados + `make ci` limpo; commits só dos arquivos tocados (nunca `git add -A`; deleções do usuário ficam fora). Atualize `docs/dev/BLOCK1-SESSION2-20261001.md`,
`LESSONS.md`, `STATE.md` e as páginas do ai-memory. Gere **novo ZIP de auditoria pequeno** (relatório, git HEAD/branch/status/log, patches aceitos,
resultados estruturados, comandos/config, testes + CI, páginas ai-memory) com `MANIFEST.tsv` (SHA256, caminho, tamanho, finalidade, artefatos referenciados excluídos).
Veredito explícito READY/NOT READY FOR BLOCK 2 com a evidência. Não pare após diagnosticar.
