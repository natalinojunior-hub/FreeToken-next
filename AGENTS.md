# AGENTS.md — Instruções para Agentes IA (freetoken-next)

**Leia primeiro:** `CONTRIBUTING.md` (vinculante para humanos e agentes). Este arquivo resume o essencial para execução de trabalho.

---

## Política IA

- Código assistido por IA é bem-vindo. **Submeter código que o contribuidor não entende NÃO é.**
- O humano por trás do PR possui cada linha, rodou em hardware real, e explica/defende sem ajuda de IA.
- Agentes **NÃO DEVEM:** `git push`, `gh pr create`, `gh pr comment`, `gh issue create` por conta do usuário.
- Agentes **NÃO DEVEM:** Escrever código, descrições de PR, ou respostas a reviewers que o usuário não compreenda totalmente.
- Agentes **NÃO DEVEM:** Reportar testes/benchmarks como executados quando não foram.
- Agente totalmente autônomo sem humano no loop → **não contribua** neste repositório.

---

## Arquivos de Contexto Vivos (Leia na Ordem)

| Arquivo | Papel | Quando Ler |
|---------|-------|------------|
| `docs/dev/CONTEXT.md` | Visão geral: projeto, hardware, checkpoints, estado atual, próximos passos | **Sempre primeiro** — orientação completa |
| `docs/dev/STATE.md` | Snapshot executável: métricas, handoffs, comandos, ambiente | Início de sessão / handoff |
| `docs/dev/ROADMAP.md` | Fases 1-18, status, gates, dependências | Planejamento / priorização |
| `docs/dev/LESSONS.md` | Padrões `sintoma → causa → fix` validados em hardware real | Debug / evitar regressões |
| `docs/dev/DECISIONS.md` | Registro imutável D-001 a D-023 (rationale arquitetural) | Entender "por que" de escolhas |
| `docs/dev/PERFORMANCE.md` | Tabelas PP/TG/VRAM/RSS por config/modelo | Validação de regressão / anchors |
| `docs/dev/ARCHITECTURE.md` | Design atual, subsistemas, seams, file:line references | Mudanças arquiteturais |
| `docs/dev/EXPERIMENTS.md` | Diário empírico EXP-000 a EXP-045 (setup/result/verdict) | Investigação profunda |
| `docs/dev/QA.md` | Gates de fase, checklists benchmark, validação conteúdo, release | Verificação / qualidade |
| `CLAUDE.md` | Ponte única: "Leia AGENTS.md primeiro" | Entrada legacy Claude Code |

---

## Layout do Repositório (Subsistemas Principais)

```
python/freetoken/      engine, package `freetoken`, CLI `ft`
  server/              OpenAI/Anthropic/Responses HTTP APIs, streaming, tool parsers
  scheduler/           chunked prefill, batching, cache manager, commit/window locking
  kvcache/             paged KV pools, radix prefix caches (radix/)
  moe/                 expert offload cache, CPU/GPU/hybrid backends, quantized experts
  models/              model registry, per-architecture loaders (qwen4_exp, glm*, deepseek_v4, gguf)
  kernel/              CUDA/Triton kernels, JIT cache, C++ extensions (csrc/)
  layers/, attention/  fused ops, attention backends (QSA, TurboKV, dense)
  engine/              cache budget planning, config resolution, VRAM ledger
  checkpoint/          HF -> FTW fast-load conversion
tests/                 mirrors python/freetoken/ by subsystem
benchmarks/            end-to-end (bench_pp_tg.py, bench_decode_moe.py) + micro
docs/                  user-facing: install, quickstart, CLI, models, ftw-hotfix
docs/dev/              developer/internal docs: architecture, state, experiments, roadmap
```

---

## Desenvolvimento

**Ambiente:** Linux x86_64, NVIDIA GPU, `uv` (não `pip` direto).

```bash
cd /models/desenvolvimento/freetoken-next
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
export TMPDIR=/models/desenvolvimento/tmp
```

**Kernels CUDA:** JIT-compilados com `nvcc` no primeiro uso (a menos que `freetoken-kernel-cache` wheel instalado).
**Extensões C++:** `python/freetoken/kernel/csrc/` built by `setup.py`; após mudanças: `python setup.py build_ext --inplace`.

**Testes:** Novo teste em `tests/` espelhando módulo protegido; estender arquivo existente antes de criar novo.
- Bug fix = teste que falha antes + passa depois.
- Mudança de performance = números A/B contra `main`.

---

## Regra Crítica: NUNCA Use RAM Como Storage

`/tmp` = **46 GiB tmpfs (RAM-backed)**. Qualquer escrita lá compete com:
- Host-resident expert banks (offload MoE ~63 GiB Flash-Next)
- PLE tables (~47 GiB)
- Pode starve/OOM-kill processo de serve não-relacionado.

**`/models` tem 1.3+ TB em disco real.** Sempre:
```bash
export TMPDIR=/models/desenvolvimento/tmp
# para: ft checkpoint, ft serve, pytest --basetemp=..., build/JIT caches, scratch downloads
```
**Pre-flight obrigatório antes de qualquer run grande:**
```bash
free -h          # RAM livre, cache limpo
du -sh /tmp/*    # lixo tmpfs de sessões anteriores
nvidia-smi       # VRAM livre, 0 processos
ps aux | grep -E '(python|tail)'  # matar órfãos/zumbis
```

---

## Workflow de Execução (Command Economy)

1. **Planeje o caminho mais barato silenciosamente** — não narre o plano.
2. **Verifique info faltando / resultado passo anterior** — pergunte se faltando, nunca assuma.
3. **Entregue exatamente o escopo pedido** — nada extra, features não solicitadas.
4. **Nunca produza output dependendo de passo não completado.**
5. **Agrupe todas as perguntas em uma mensagem.**

**Ferramentas — Ordem de preferência (token economy):**
1. `grep` > `read` (narrow query primeiro)
2. `read` line ranges, nunca arquivos inteiros
3. Nunca re-leia arquivo já lido na sessão
4. Shell output sempre filtrado: `head/tail/wc/grep`
5. Mínimo tool calls para certeza — pare quando certo

---

## Commits

[Conventional Commits](https://www.conventionalcommits.org/), uma linha, imperativo, lowercase, sem ponto final:

```
fix(kvcache): size the SWA radix pool for chunked prefill
feat(mtp): add carry shift for draft alignment
perf(qsa): fuse in-sram fp4 dequant
```

PRs são squash-merged; título do PR segue mesmo formato.
**Só commit quando o usuário pedir.** Atribuição: `Assisted-by: <agent name>`.

---

## Comandos de Verificação (Rodar Antes de Entregar)

```bash
# Lint / Typecheck (se configurado)
# uv run ruff check .       # ou equivalente do projeto
# uv run mypy python/freetoken  # ou equivalente

# Testes rápidos
uv run pytest tests/ -m "not slow" -q --basetemp=/models/desenvolvimento/tmp

# Benchmark guard (se mudando performance)
uv run python benchmarks/bench_pp_tg.py --model /models/Qwen3.6-35B-A3B-NVFP4-FT \
    --tokens 16384 --decode 128 --repeats 3 --label guard \
    --serve-arg "--num-tokens 16576" --serve-arg "--cache-type naive" \
    --json /models/desenvolvimento/tmp/guard.jsonl
```

---

## Issues / PRs

- Busque issues/PRs existentes antes de começar. [Roadmap](https://github.com/FlashML-org/FreeToken/issues/79) discutido com maintainers antes de implementar.
- Draft issue: use template `.github/ISSUE_TEMPLATE/` (engine bug, model checkpoint, feature request) — preencha **todos** campos obrigatórios: hardware, driver, versão FreeToken, checkpoint ID, comando exato, log completo.
- Uma mudança por PR, linkada à issue, com hardware/checkpoint/comando testado.

---

## Code Comments

- Explicam "why" não-óbvio, nunca repetem o código.
- Escreva código primeiro, comente só onde leitor ficaria confuso.
- Máximo 1-2 linhas. Arquivos de config = sem comentários.
- ASCII: `-` não em-dash, `->` não setas.

---

## Referências Rápidas (File:Line Patterns)

| Área | Arquivos-Chave |
|------|----------------|
| VRAM Ledger | `engine/vram_ledger.py`, `engine/cache_budget.py` |
| TurboKV/QSA Split | `kvcache/qsa_pool.py`, `kvcache/qsa_sparse.py`, `kernel/triton/qsa/decompress.py` |
| MoE Offload | `moe/offload_cache.py`, `moe/expert_banks.py` |
| MTP Spec | `scheduler/spec.py`, `models/qwen4_exp/model.py` |
| GGUF Loader | `models/gguf/reader.py`, `models/gguf/config.py`, `layers/gguf.py` |
| Chunked Prefill | `scheduler/prefill_adder.py`, `scheduler/cache.py` |
| Radix Cache | `kvcache/radix/radix_cache.py` |

---

## Próximas Ações Prioritárias (Contexto Atual)

1. Validar split-kernel Turbo4/QSA 16K (sha1 match baseline)
2. Medir TG + accept-rate MTP=1 com Turbo4 ativo
3. Debug non-determinismo sequential requests (log logits top-2)
4. Live test k=2/k=3 content equivalence
5. Phase 7: Expert pool keyed by (bank, role, type)

> **Detalhes completos em:** `docs/dev/CONTEXT.md` → "Próximos Passos Imediatos" e `docs/dev/STATE.md` → "Handoff Crítico"