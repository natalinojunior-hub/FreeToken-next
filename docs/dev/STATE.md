# STATE — freetoken-next

**Doing:** Passos 1-5 do plano executados. `cert_matrix.py --contexts 16384` rodou; 1 regressão de guard + 2 falhas pré-existentes achadas (não causadas pelo fix de hoje).

**Done (2026-09-20):**
1. CUDA graph fix (`ba7f976`) reconfirmado: sem hang, 2 repeats de 256 tok, TG 33.10.
2. Regressão real 0.1.2 vs HEAD: **-12.3%** (34.7 -> 30.44 tok/s, sha1 idêntico). Causa não isolada — candidato: VRAM ledger mem-ratio 0.98 (HEAD) vs ~0.9 (0.1.2).
3. Crash k=3 (`spec.py:355`) corrigido em `gdn.py` (`.squeeze(0)` na captura do recurrent_states). Suite mínima 21/21 passando.
4. GGUF Unsloth-IQ4_XS: 3 bugs corrigidos em `gguf.py` — indexer `index_kv_heads`, PLE `ple_layer_index`, e `ModelConfig` faltando `slot_states=ple_slot_states(qwen4_args)` (causava `PLE needs ple_ngram_ctx slot state`). Diff de kwargs `parse_config` vs `gguf.py` ModelConfig confirmado limpo (únicas ausências são `image_token_id`/`vision_config`, default None, GGUF não é multimodal). **Verificado end-to-end:** 4096 tok prompt / 32 decode (31/32 gerados, EOS antecipado) via `--no-graph --cache-type=radix`, TG 31.36 tok/s — **não comparável** aos números de 33.10/30.44 (profundidade, decode length e graph diferentes); confirma só que o path GGUF funciona, não sua velocidade relativa.

5. `cert_matrix.py --contexts 16384` (7 rows, 3 já BLOCKED por geometria mista de experts, pré-existente):
   - `native-35b-a3b`: **REGRESSION** de guard — PP 4575.8 < 4600.0 (-0.5%); TG 158.63 passou. Guard desatualizado ou regressão real de PP não investigada.
   - `native-flash-next`: FAIL — `AssertionError: cache budget too small` a mem-ratio 0.86/16384 ctx (GPU livre, não é contenção externa). Pré-existente, não causado pelo fix de hoje.
   - `gguf-flash-unsloth-ud`: FAIL por **stall-timeout de prefill (90s)**, não pelo bug do PLE (esse já foi confirmado corrigido isoladamente a 4096 ctx, TG 31.36). Causa: log mostra chunk de 8192/16384 tokens levando ~41s (PP ~194 tok/s, sem kernel MMQ para IQ4_XS) — 2 chunks ultrapassam os 90s do watchdog do bench, não é deadlock.
   - `gguf-qwen38-27b-iq3s`: FAIL — `CUDA out of memory` real (dense I-quant, mem-ratio 0.86 insuficiente a 16384 ctx).

**Decisões:** Métrica soberana = TG (tok/s). Só 2 modelos autorizados (AGENTS.md).

**Next:** Investigar a regressão de PP em `native-35b-a3b` (-0.5% no guard) e decidir se o guard de PP precisa recalibração; separadamente, `gguf-flash-unsloth-ud` precisa de `--stall-timeout` maior ou `--decode`/contexto menor para caber no cert_matrix a 16384 tokens (limitação do harness, não do modelo).
