# STATE — freetoken-next

**Doing:** Auditoria FreeToken vs LTO (TG) no RTX 5080 — encerrada por ora. Premissa original ("comparação NVFP4+Turbo4 justa vs LTO") invalidada: LTO é GGUF-only, não roda o checkpoint NVFP4-Radix. Ver PERFORMANCE.md "Correção de premissa" para os números não-comparáveis disponíveis dos dois lados.

**Done:**
- Kernel cache prebuilt estava faltando (arquivo deletado) — restaurado e rebuildado via `scripts/build-release-wheels.sh`.
- WIP não commitado do MoE offload trava indefinidamente no caminho de CUDA graph decode (qualquer ctx); `--no-graph` contorna. Diagnosticado com py-spy (processo-pai, `-s --native`) até localizar em `torch/cuda/graphs.py replay()`; mecanismo exato não confirmado, testes de env var (`FUSED_COPY=0`, `SMALL_BANK_FEAT_BYTES=262144`) não resolveram. Ver LESSONS.md.
- Baseline eager válido obtido: **TG 23.21 tok/s** @ 16384 ctx, `--no-graph`, mem-ratio 0.85, 1089 slots de expert cache (mem-ratio 0.9 dá OOM real, não fragmentação — `expandable_segments` testado e não ajuda).
- Varredura MTP NVFP4 k=0..6 (via CUDA graph, hoje quebrada no WIP): k=0 TG 27.73, k=1 TG 24.39, k=2 TG 25.09 (melhor MTP, ainda < baseline), k=3 crash (GDN shape mismatch). GGUF Unsloth-IQ4_XS bloqueado (PLE/indexer dims incompatíveis).

**Decisões:** Métrica soberana = TG (tok/s), não taxa de aceitação. Só 2 modelos autorizados (ver AGENTS.md). Working tree tem muitos arquivos modificados não commitados (docs/dev, benchmarks, kernels) — não commitar sem pedido explícito.

**Achado principal desta sessão:** regressão real de ~31% intra-FreeToken confirmada — 0.1.2 fazia 34.7 tok/s @ ctx=8192 no NVFP4-Radix; build atual faz 23.94 tok/s no mesmo ctx=8192 (controle direto, não é efeito de escala de contexto: 16K dá 23.21, quase igual). Não é FreeToken perdendo para LTO — é uma regressão entre versões do próprio FreeToken. Ver PERFORMANCE.md.

**Next:**
1. Bisectar commits entre 0.1.2+gaf71ba432 e HEAD para achar onde os ~31% de TG foram perdidos (candidatos: mudanças no MoE offload/host-bank cache, dado que RSS 70GB e o host-bank system parecem ter crescido desde então).
2. Achar causa raiz do hang em CUDA graph replay() (precisa cuda-gdb/nsys — py-spy não vê além do pybind/driver).
3. Investigar crash k=3 (GDN shape mismatch) em `scheduler/spec.py` verify forward.
4. Resolver mismatches GGUF Unsloth-IQ4_XS (PLE dims, indexer heads).
5. Rodar `benchmarks/cert_matrix.py`.
