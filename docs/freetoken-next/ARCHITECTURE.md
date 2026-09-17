# ARCHITECTURE — freetoken-next

Current truth about the design. Sections 1–2 describe the inherited v0.1.3 base as it
exists; sections 3+ are the target design and are filled in as each subsystem lands
(audit evidence is summarised here, raw detail stays in the audits/EXPERIMENTS.md).

## 1. Inherited subsystem map (upstream v0.1.3, `cac247a`)

```
python/freetoken/            engine, installed as `freetoken` with the `ft` CLI
  server/                    OpenAI / Anthropic / Responses HTTP APIs, streaming, tool parsers
  scheduler/                 chunked prefill, batching, cache manager, paged-KV reservation
  kvcache/                   paged KV pools + radix prefix caches (radix/, DSV4 pools, rebuild)
  moe/                       expert offload cache, CPU/GPU/hybrid executors, quantized experts
  models/                    model registry + shared loading machinery (sharding, qkv/expert
                             merge, streaming layers into banks)
  kernel/                    CUDA/Triton kernels, JIT cache, C++ exts in csrc/
                             (_pinned_tensor, _cpu_moe, _ple_store)
  layers/, attention/        fused ops; backends triton / fa / fi(flashinfer) / trtllm /
                             dsv4_sparse / dsa / linear
  engine/                    cache budget planning, config resolution, backend gating
  checkpoint/                HF -> FTW fast-load conversion (incl. Flash-Next PLE table)
  benchmark/, shell/, daemon/, distributed/, llm/, message/, mm/, tokenizer/, utils/
tests/                       mirrors the subsystem tree (see tests/README.md)
benchmarks/                  bench_decode_moe.py, bench_load_weight_generic.py,
                             bench_offload_cache_copy.py
freetoken-kernel-cache/      companion wheel of prebuilt kernels
```

Auto-resolution is the product's core promise: `ft serve --model X` alone resolves dtype,
attention backend, MoE strategy, MoE cache size, KV capacity, CUDA-graph sizes and parsers
from the checkpoint + GPU. Memory surface today: `--memory-ratio` (default 0.9 of free VRAM
for weights + MoE cache + KV), `--num-pages`/`--num-tokens`, `--page-size`, `--cache-type
radix|naive`, `--moe-cache-size|-rate|-auto`, `--kv-reserve-tokens` (8192 floor reserved
before expert cache fills), `--moe-prefill-hit-d2d` (off), `--moe-cpu-layers`.
Introspection: `GET /v1/stats` (throughput, latency, VRAM, pool occupancy),
`GET /v1/cache/status` (pool table), `POST /v1/cache/rebuild` (live pool resize).

Quant surface today: `--quant-backend layer[.kind]=name` (`linear=marlin`, `moe=b12x`,
`moe.nvfp4=triton`), KV dtype comes from the checkpoint (`hf_quant_config.json`
`kv_cache_quant_algo`, e.g. FP8 for the local Qwen3.6-35B-A3B-NVFP4), with the
`QuantConfig → QuantScheme → QuantMethod` layers from `#418` underneath.

## 2. What the base already gives the mission

- Native GGUF: `gguf>=0.19,<1` is a core dependency, but a loader is **not** confirmed
  present yet (audit A1/A4 verdict pending).
- Paged KV pools with a rebuild path (`ft ctl cache --kv N`) — the handle tiering will use.
- One device-side LRU admission kernel for the expert cache (`flashlib`), i.e. expert
  residency is already governed by a kernel-level policy, not only Python.
- `_pinned_tensor` C++ extension — pinned host memory primitive for RAM tiers.
- `_ple_store` disk row store — the pattern for "cold tier backed by something slower".
- CPU MoE executor with AVX-512 bf16/nvfp4 microkernels + runtime dispatch.

## 3. Native GGUF (Phase 2)

_Pending audit A4 verdict; will record: tensor-table representation, name translation to
engine roles, per-type dispatch (DIRECT / REPACK-ONCE / NEW-KERNEL / CPU-ONLY), the
derived-overlay location, sharding, tied embeddings, and the first dense increment._

## 4. KV quantization: Turbo3 / Turbo4 / TCQ / VBR (Phases 3–4)

_Pending audits A1 (host abstraction) and A2 (reference format)._

## 5. Tiered / paged KV and the VRAM governor (Phases 5–8)

_Pending audit A1 §6._

## 6. MTP / speculative decoding (Phases 9–10, 12)

_Pending audit A3._
