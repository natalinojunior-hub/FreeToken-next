# State Snapshot — 2026-09-22

## Doing Now
- MTP usa banco próprio blk.48 (Q8_0) no cache de offload (banco 48, `ModelConfig.mtp_expert_bank`). Sem commit.

## Done
- Aceitação k=1 55.5 % -> 75.2 %. k=1 TG 22.43 -> 22.47 (sem ganho): verify 59.3 -> 69.9 ms, replay 15.5 -> 20.1 ms, draft 3.2 -> 3.8 ms.
- Causa da piora de verify/replay: gate_up Q8_0 do MTP cria pool GPU extra de cache_size slots (per-geometria) -> expert slots 1181 -> 738 -> mais misses.
- k=0 35.30 (inalterado, mesma sha). make ci PASS (1973).
- Estudo: `relatorio-estudo-mtp-lto.md`.

## Next
1. Pool por geometria dimensionado pela demanda (banco único de 1 camada não deve custar cache_size slots) — toca planner/offload_cache; pedir aval.
2. Verify em CUDA graph. 3. Zero-replay GDN+QSA+PLE.
- Commit só quando o operador pedir.
