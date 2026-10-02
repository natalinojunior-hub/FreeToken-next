# FreeToken-next — relatório da campanha de consolidação (2026-10-02)

| item | resultado |
|---|---|
| HEAD final | `6b070c1` (branch `next`, base auditada `e7b4d31`) |
| commits | 9 (15 arquivos, +332/−41): `cf26a2f` `c070c71` `fa5802d` `6356685` `490f572` `8fb221e` `f9a3850` `b0863d2` `6b070c1` |
| paridade AD | FECHADA. Causa: conv curta do PLE no verify via cuDNN ≠ soma fp32 do decode (`L1.ple.out0`, não camada 7). 16K prompts 0/3/7 × k0–k5: 18/18 SHA == RAW; revalidado p3 após os commits seguintes (6/6) |
| OOM especulativo | PARCIAL. OOM antes do verify: rollback, encolhe experts, repete a profundidade (2 injeções + recuperações reais, SHA == RAW, 0 requests perdidos). Reserva aprendida passou a persistir (antes nunca salvava). Aberto: OOM dentro do verify zero-replay (1 ocorrência, não reproduzida) |
| MTP nos 35B oficiais | HABILITADO com prova. Causa da divergência: linears de 2..8 linhas (FP8 per-tensor e NVFP4) ≠ M==1. Correção: GEMV multi-linha com aritmética M==1 (técnica do Strata). Gate p0/p3/p7 × k0–k5: 18/18 nos três modelos |
| bug de segurança | GGUF `qwen35moe` escapava do bloqueio de MTP sem paridade (servia saída divergente) — fechado |

## Modelos oficiais (16K, decode 256, graph on, 3×RAW + 3×auto, SHA auto == RAW)

| modelo | caminho | RAW TG | MTP TG | PP 16K (sweep) |
|---|---|---|---|---|
| Qwen3.6-35B-A3B-UD-NVFP4-Fast | `/models/Qwen3.6-35B-A3B-UD-NVFP4-Fast` (23,69 GB) | 157,7–158,9 | 185,1–190,9 (+19%) | 2988 com MTP; 4231 sem MTP |
| Ornith-1.5-35B-A3B-NVFP4 | `/models/Ornith-1.5-35B-A3B-NVFP4` (23,46 GB) | 123,9–124,4 | 147,5–147,8 (+19%) | 2521 com MTP (medido antes de `6b070c1`) |
| Ornith-1.5-35B-A3B GGUF | `/models/Ornith-1.5-35B-A3B-APEX-MTP-I-Compact.gguf` (17,44 GB) | 152,2–152,9 | 190,8–195,7 (+27%) | ~45k (prefixo em cache; PP frio não medido) |

Observação: com MTP ligado o RAW do Ornith NVFP4 cai de 133–135 para 124 (VRAM do MTP sai dos experts); o modo auto ainda ganha +10% sobre o RAW antigo.

## Strata / papers / DFlash

| candidato | aplica agora? | decisão |
|---|---|---|
| Strata: redução de M linhas com topologia M==1 | sim | ACEITO (`8fb221e`, `f9a3850`) |
| Strata: verify recorrente read-only + cópia da linha aceita | já equivalente (zero-replay + `spec_states`) | sem mudança |
| Strata: padrões Windows (CreateFileMappingW, cudaHostAlloc mapped, Job Objects, %APPDATA%) | referência | não implementado nesta sessão |
| HySparse2 (arXiv 2609.26368) | não: exige pré-treino | REJEITADO |
| DeepSeek-V4.1-Flash CED/CSA2 (arXiv 2609.19969) | não: exige pré-treino | REJEITADO |
| DFlash (drafters locais `dflash`, 6 camadas, bloco 16) | tecnicamente sim | NO-GO por ora: em offload de experts o custo do verify cresce com linhas (k5 < k1–k3 nos 3 modelos), bloco 16 seria pior; drafter de 0,24–0,78 GB tira experts da VRAM; FreeToken não tem o caminho DFlash |

## Startup (boot quente, calibração em cache)
- Ornith NVFP4 ≈ 17 s até pronto: init 4 s · pesos 7 s · planner + Phase I 4 s · grafos 1–2 s.
- Ornith GGUF ≈ 23 s (bancos de experts em caminho serial: 6 s).
- Qwen3.6 com MTP ≈ 19–20 s; a validação Phase I estoura em todo boot (+194 MiB não precificados) e o re-solve tira 0,53 GiB dos experts.
- Chave dos perfis inclui o hash do código-fonte: qualquer commit invalida reserva e calibração inteiras (invalidação total, não por campo).
- Meta 5–10 s e redesign de perfil/`--bench`: NÃO feitos.

## Perfil / `--bench`
- Hoje: perfis separados (VRAM aprendida + calibração, profundidade MTP, `ft tune`, `ft bench bw`); não existe `--bench`. Correção desta sessão: a reserva aprendida de VRAM agora é carregada/salva de fato.

## Linux / Windows
- Linux: validado. Windows: nada implementado nem executado; sem afirmação de paridade.

## Certificação
- 16K: matriz oficial acima (3×RAW + 3×MTP por modelo, SHA == RAW). PP frio/quente e qualidade ainda não padronizados → 16K NÃO certificado formalmente.
- 262K: não executado (a pilha 16K ainda mudou nesta sessão).

## CI
- `make ci` em worktree limpo no HEAD final `6b070c1` (com as extensões nativas `.so` copiadas): **2783 passed, 0 failed, 209 skipped, 17 deselected (slow)**; Ruff ok; MyPy ok (447 arquivos).

## Bloqueios / próximos passos
1. Phase I do Qwen3.6 com MTP: achar o dono de +194 MiB (backend criado depois do snapshot + resíduo do chunk) e precificar.
2. OOM dentro do verify zero-replay: snapshot barato do slot linear ou reserva do verify medida.
3. PP com MTP nos NVFP4 (−30%): o preenchimento do KV do draft roda o prompt em janelas de 8 linhas.
4. Perfil único com invalidação por campo + `--bench`; boot quente 5–10 s.
5. Camada de plataforma Windows.
