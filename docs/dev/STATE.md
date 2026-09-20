# STATE — freetoken-next

**Doing:** Investigação de regressão de TG entre FreeToken 0.1.2 e HEAD — pausada. Premissa de comparação FreeToken vs LTO já estava invalidada (LTO é GGUF-only). Agora também invalidada a comparação intra-FreeToken 0.1.2 vs atual (ver Achados).

**Done:**
- Baseline eager válido: **TG 23.21 tok/s** @ 16384 ctx, `--no-graph`, mem-ratio 0.85, 1089 slots de expert cache.
- Varredura MTP NVFP4 k=0..6: k=0 TG 27.73, k=1 TG 24.39, k=2 TG 25.09, k=3 crash (GDN shape mismatch). GGUF Unsloth-IQ4_XS bloqueado (PLE/indexer dims incompatíveis).
- Causa raiz de um crash de captura de CUDA graph identificada e corrigida (commit `ba7f976`): `qsa_sparse.py` fazia `indices[indices >= 0]` (boolean-mask, shape dinâmica) dentro do stream de captura -> `cudaErrorStreamCaptureInvalidated`. Fix: guarda `torch.cuda.is_current_stream_capturing()` pulando a seleção de páginas durante captura. Não confirmado se resolve 100% do hang em replay() já documentado — precisa reteste.
- Snapshot de segurança commitado: `ba7f976` (470 arquivos, todo o WIP de MoE offload + docs).

**Achados desta sessão (correção de achado anterior):** a suposta "regressão de ~31%" (0.1.2 34.7 tok/s vs atual 23.94 tok/s, ambos @ "ctx=8192") **não é uma comparação válida**. Releitura do `stats.json`/`ServerArgs` do boot 0.1.2 mostrou: `max_seq_len_override=4096`, prompt de 22 tokens, 31 completions — decode com KV quase vazio, não "8192 preenchido" (`kv_reserve_tokens=8192` é reserva de VRAM, não profundidade usada). 0.1.2 também não tem campo kv-format/turbo4 (KV puro, `cache_type='radix'`) e rodou com CUDA graph ligado; a rodada atual usada na comparação foi `--no-graph` e possivelmente turbo4. Reproduzi o boot 0.1.2 exatamente (mesma venv pinada em 0.1.2+gaf71ba432) e bati o número: **34.7 tok/s de novo, idêntico** — confirma que o número antigo é real e reprodutível *para aquela condição*, mas não é comparável ao número atual. Nenhuma regressão foi de fato medida ainda. Ver PERFORMANCE.md.

**Decisões:** Métrica soberana = TG (tok/s), não taxa de aceitação. Só 2 modelos autorizados (ver AGENTS.md).

**Next:**
1. Reteste do caminho CUDA graph com o fix de `qsa_sparse.py` já commitado — confirmar se o hang em replay() sumiu ou se há causa adicional.
2. Só depois disso, medir uma regressão real: mesmo commit, mesma profundidade de KV preenchida, mesmo kv-format, mesmo caminho graph/eager entre 0.1.2 e HEAD. Bisect só faz sentido com esse baseline válido em mãos.
3. Investigar crash k=3 (GDN shape mismatch) em `scheduler/spec.py` verify forward.
4. Resolver mismatches GGUF Unsloth-IQ4_XS (PLE dims, indexer heads).
5. Rodar `benchmarks/cert_matrix.py`.
