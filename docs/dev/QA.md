# QA.md — Quality Assurance Gates & Checklists

**Projeto:** freetoken-next | **Hardware:** RTX 5080 16GB | **Base:** FreeToken v0.1.3 (`cac247a`)

---

## Gates de Fase (Roadmap)

| Fase | Gate | Comando Validação | Critério Pass |
|------|------|-------------------|---------------|
| 1 | Lineage atual | `git fetch upstream && git rev-parse HEAD upstream/main` | 0 commits behind |
| 2 | Build limpo | `uv pip install -e ".[accel]" && ft --version` | 3 C++ exts compilados, versão 0.1.3 |
| 3 | Baselines | `bench_pp_tg.py --tokens 16384 --decode 128 --repeats 3` | 35B: PP≥4600 TG≥158; Flash: PP≥1850 TG≥28.5 |
| 4 | Audits A1-A9 | `ls docs/freetoken-next/audits/A*.md` | 9 arquivos, integrados em ARCHITECTURE.md |
| 5 | VRAM Ledger | `ft serve --model ... --memory-ratio 0.9` | Flash-Next serve sem OOM; account printed |
| 6 | 128K/256K | `bench_pp_tg.py --tokens 131072/262144` | 35B: 128K TG≥89, 256K TG≥63; VRAM≤15.51 GiB |
| 7 | GGUF MoE Geometry | Pool keyed by (bank, role, type) | Ornith/Tiel MoE GGUF carregam |
| 8 | Turbo4+MTP Cert | `bench_pp_tg.py --spec-mtp 1 --kv-format turbo4` | sha1 `614aa7bcdf59` match baseline; TG≥19 |
| 9 | MTP qwen4exp | `bench_pp_tg.py --spec-mtp 2 --decode 64` | 2.82 tok/step, 86.4% 2/2, sha1 `ed45eb6cc897` |
| 10 | MTP+TurboKV Fusion | Split-kernel path validated | Decompress kernel separate; no ptxas OOM |
| 11 | TCQ/VBR | Per-(layer,side) tier schedule | Gates Turbo honrados |
| 12 | PLE Tiered RAM | `--kv-reserve-context` + PLE streaming | Bounded-read sparse/windowed |
| 13 | Adaptive MTP | `--context auto` functional | TG model survives expert cache |
| 14 | 512K/1M | RoPE-scaled checkpoint | KV affordable only compressed |
| 15 | Cert Matrix | `benchmarks/cert_matrix.py` | All native -FT rows clear guard + GGUF parity |
| 16 | Matrix Gaps | Shard join, qwen4exp GGUF adapter, PLE mapping | All Flash GGUFs unblocked |

---

## Test Suite Gates

```bash
# Suite rápida (pre-commit / CI)
pytest tests -m "not slow" -q --basetemp=/models/desenvolvimento/tmp
# Gate: 1839 passed, 206 skipped, 1 failed (flashinfer fp4_quantization_120f - env)

# Suite completa (pre-release)
pytest tests -q --basetemp=/models/desenvolvimento/tmp
# Gate: 0 failures regressivos (exceto flashinfer conhecido)
```

**Regra:** Fix em código compartilhado (cache/paginação/scheduler) -> rodar **suite completa do subsistema**, não só arquivos tocados.

---

## Benchmark Validation Checklist

### Pre-Flight (Obrigatório)
- [ ] `free -h` — RAM livre ≥ 20 GiB, swap 0, tmpfs limpo
- [ ] `nvidia-smi` — VRAM livre ≥ 14 GiB, 0 processos
- [ ] `ps aux | grep -E '(python|tail)'` — 0 órfãos
- [ ] `du -sh /tmp/*` — < 1 GiB lixo
- [ ] `export TMPDIR=/models/desenvolvimento/tmp`

### Execução
- [ ] `--cache-type naive` (prefix cache faking PP)
- [ ] `--num-tokens` explícito = piso obrigatório (não sugestão)
- [ ] `--memory-ratio` testado: 0.9 (default), 0.86 (hand-tuned), 1.0 (ceiling)
- [ ] `--cuda-graph-max-bs 0` para MTP (overlap disabled)
- [ ] `FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1` exportado ANTES do python
- [ ] Mínimo 3 repeats, warmups declarados
- [ ] JSONL output salvo para auditoria

### Pós-Execução
- [ ] `sha1` comparado entre runs (determinismo)
- [ ] `sha1` comparado 2 processos servidor separados (state leak check)
- [ ] VRAM peak ≤ 15.51 GiB (RTX 5080 ceiling)
- [ ] RSS host ≤ 84 GiB (91 GiB total - 7 GiB system)
- [ ] GPU util ≥ 95% (compute-bound, não memory-bound)
- [ ] Logs `spec.py:195` capturados (acceptance/drafts/sampled)

---

## Content Equivalence Gates (MTP)

| Teste | Prompt | Critério |
|-------|--------|----------|
| Short | "Hello world" | sha1 match `--spec-mtp 0` |
| Long | "ocean poem" (pegou k=1 bug) | sha1 match `--spec-mtp 0` |
| Multi-token k=2 | Short + Long | `accepted=2/2`, sha1 match |
| Multi-token k=3 | Short + Long | `accepted=3/3`, sha1 match |
| 2 processos isolados | Same prompt | sha1 idênticos cross-process |

---

## Regression Anchors (Não Quebrar)

```bash
# 35B-A3B 16K naive cache
PP ≥ 4600 tok/s
TG ≥ 158 tok/s
VRAM ≤ 15.0 GiB
sha1: 614aa7bcdf59 (Turbo4+MTP) / baseline triton+bf16

# Flash-Next 16K naive cache
PP ≥ 1850 tok/s
TG ≥ 28.5 tok/s
VRAM ≤ 15.0 GiB
sha1: 614aa7bcdf59 (Turbo4+MTP) / baseline triton+bf16

# Dummy-page brick
Pool allocates num_pages + 1
Both budget formulas price it
35B @16K re-measured: PP 4610.1 / TG 158.75, sha1 identical
```

---

## Code Quality Gates

- **Type hints:** Novas funções públicas tipadas
- **Docstrings:** Públicas com args/returns/exceptions
- **Tests:** Bug fix = teste que falha antes + passa depois
- **Performance:** Mudança de perf = A/B numbers contra main
- **No RAM storage:** `/tmp` = tmpfs = RAM real -> use `TMPDIR=/models/desenvolvimento/tmp`
- **Conventional Commits:** `fix(scope): msg` | `feat(scope): msg` | `perf(scope): msg`

---

## Release Checklist (Cert Matrix Row)

- [ ] Native `-FT`/NVFP4 row clears guard (PP/TG/VRAM)
- [ ] Same-arch GGUF row reports parity vs native
- [ ] Blocked rows name blocker (não deletados)
- [ ] `benchmarks/cert_matrix.py` green
- [ ] `CHANGELOG.md` updated (se existir)
- [ ] `ft --version` bump (se release tagged)

---

## Debug Checklist (Non-Determinismo / State Leaks)

- [ ] Log logits top-2 em `spec.py:195` (não só argmax)
- [ ] Run `--decode 4 --repeats 6` + diff phase-1 vs phase-2 logs
- [ ] Compare 2 isolated server processes (1 request each)
- [ ] Check `model.model._last_residual` overwrite pattern
- [ ] Check `req.uid` uniqueness per real request
- [ ] Verify `free_spec_snapshot_slot` called in `_free_req_resources`
- [ ] Verify `QSAKVCache.free_req(table_idx)` zeros `pending_ring` + `_cmp_k_buffer`