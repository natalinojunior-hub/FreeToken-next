# State Snapshot — 2026-09-20

## Doing Now
- Investigar gargalo PCIe/heterogeneidade no Flash-Next NVFP4-Radix (decode TG)

## Done This Session
1. Refutada premissa "LTO supera FreeToken": LTO só roda GGUF (não NVFP4-Radix), número 38 tok/s é IQ4_XS prompt curto com MTP n_max=3 não certificado
2. Testado `--moe-strategy hybrid` (CPU+GPU co-compute, nunca usado antes no Flash-Next): +3% TG (29.61→30.51), bit-idêntico, auto-tune 48.7% PCIe/51.3% CPU
3. Conclusão: PCIe expert-streaming NÃO é o gargalo dominante (~poucos % do step); 35B-A3B faz 6.3ms/token vs Flash-Next 33ms/token pela residência de experts em VRAM, não por PCIe
4. Testado `--ple-backend pinned`: falhou por RAM insuficiente (RSS já ~70 GiB em `disk`)
5. Docs atualizados em `docs/dev/PERFORMANCE.md`

## Decisions
- `--moe-strategy hybrid` candidato a default no Flash-Next (ganho grátis, sem risco de correção)
- Meta de 60 tok/s no Flash-Next não é alcançável só por otimização de decode/PCIe — exigiria experts residentes em VRAM ou checkpoint menor

## Next Steps
1. Aguardando decisão do usuário: (a) tornar hybrid default, ou (b) investigar path GDN/QSA decode / dequant NVFP4 via profiling
2. Regressão interna não resolvida: bisect 34.7 (v0.1.2) → 30.44 (HEAD) tok/s, mesma config isomórfica
