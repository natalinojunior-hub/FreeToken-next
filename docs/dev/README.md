# Documentação Interna & Dev — freetoken-next

**Última atualização:** 2026-09-18 | **Versão:** FreeToken v0.1.3 (`cac247a`)

Esta pasta contém o Single Source of Truth do estado do desenvolvimento.

## Arquivos de Contexto Vivos

| Arquivo | Descrição |
|---------|-----------|
| `CONTEXT.md` | **Visão geral completa**: projeto, hardware, estado, próximos passos. Leia sempre primeiro. |
| `STATE.md` | Snapshot executável: métricas atuais, handoffs e comandos críticos. |
| `ROADMAP.md` | Fases 1-18, gates e dependências para rastreabilidade de entrega. |
| `LESSONS.md` | Padrões empíricos de `sintoma → causa → fix` validados no hardware real. |
| `DECISIONS.md` | Registro imutável de decisões arquiteturais (D-001 a D-023). |
| `PERFORMANCE.md` | Tabelas de PP/TG/VRAM/RSS por config/modelo (Regression Anchors). |
| `ARCHITECTURE.md` | Mapas de subsistemas, design de módulos e referências críticas (file:line). |
| `EXPERIMENTS.md` | Diário de experimentos controlados (EXP-000 a EXP-045). |
| `QA.md` | Checklists de validação e gates de release (Quality Assurance). |

---

## Documentação de Usuário (docs/)

Os arquivos voltados para o usuário final estão na raiz de `docs/`:
- `install.md`, `quickstart.md`, `models.md`, `cli.md`, `ftw-hotfix.md`.

---

## Referência Rápida (para uso com Makefile)

```bash
# Rodar todos os testes rápidos
make test

# Rodar linter (Ruff)
make lint

# Formatar código
make format

# Executar benchmark baseline de regressão
make bench
```
