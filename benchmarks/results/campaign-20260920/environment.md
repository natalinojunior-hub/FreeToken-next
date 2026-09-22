# Campaign Environment — 2026-09-20

## FreeToken Repository
- Path: `/models/desenvolvimento/freetoken-next`
- Commit: `20ca91764feab5ed244775546c2caf74bfacebd8`
- Message: `docs: update living docs for MTP speculative decode optimization`
- Local changes:
  - M docs/dev/LESSONS.md
  - M docs/dev/PERFORMANCE.md
  - M docs/dev/STATE.md
  - M python/freetoken/models/qwen4_exp/model.py
  - ?? goal-20260920-2037.txt

## LTO Repository
- Path: `/models/servers/llama-turbo-optimal`
- Key reference document: `QWEN38_FLASH_MTP_PERFORMANCE.md`

## Hardware
- **GPU**: NVIDIA GeForce RTX 5080 (16303 MiB / 15.51 GiB VRAM)
  - Driver: 610.57.04
  - CUDA: 13.3 (nvcc 13.3.73)
- **CPU**: AMD Ryzen 9 9900X 12-Core Processor (24 threads)
- **RAM**: 91 GiB total, 81 GiB free, 3.6 GiB cache, 0 Swap

## Software
- **FreeToken version**: 0.1.3
- **Python**: 3.12 (via uv venv)
- **TMPDIR**: `/models/desenvolvimento/tmp`

## Concurrent Processes (at snapshot)
- systemd services (networkd-dispatcher, tailscaled, unattended-upgrades)
- headroom-ai agent (2 processes)

## Models Allowed (per goal)
1. `/models/Qwen3.8-Flash-Next-NVFP4-Radix` — Native NVFP4, GDN+QSA+MTP on SM120
2. `/models/Qwen3.8-Flash-Next-Unsloth-IQ4_XS/UD-IQ4_XS` — GGUF IQ4_XS sharded + MTP heads in `MTP/`

## Key LTO Historical Configuration (from QWEN38_FLASH_MTP_PERFORMANCE.md)
- **Target model**: Qwen3.8 Flash Next IQ4_XS
- **MTP head**: shared-Q8 (primary), shared-Q4 (secondary)
- **KV format**: Turbo4 target, q4_0 draft KV (q8_0 as control)
- **MTP depth**: n_max=3 (also tested n_max=1,2)
- **Fit policy**: `--fit off` (manual placement) — **critical**: `--fit on` (auto-fit) typically gives worse results; explicit `--n-cpu-moe` and `--spec-draft-n-cpu-moe` required with `--fit off`
- **GPU layers**: `--ngl auto --ngld auto` (but overridden by explicit CPU-MoE placement when `--fit off`)
- **Single stream**: `-np 1`
- **Prompt**: Code prompt (4096 tokens effective context)
- **Measured result**: ~52-56 tok/s median TG with MTP n_max=3 auto-fit q4_0 (but best verified results use `--fit off` with explicit placement)

## FreeToken Equivalent Flags (to map)
- `--kv-format turbo4` → Turbo4 target KV
- `--spec-mtp 3` → n_max=3
- `--moe-cache-auto` → auto-fit MoE cache
- `--cache-type naive` → naive cache (or radix if GDN-aware needed)
- `--moe-strategy hybrid` → CPU+GPU co-compute (tested +3% TG in Flash-Next)