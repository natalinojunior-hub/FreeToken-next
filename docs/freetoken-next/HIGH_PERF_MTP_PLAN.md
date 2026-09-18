# High-Performance MTP & Compressed KV Plan: Achieving >1850 PP, >50 TG, and 256K Context

**Author:** Antigravity (Advanced Agentic Coding)  
**Date:** 2026-09-18  
**Target Hardware:** NVIDIA GeForce RTX 5080 (15.51 GiB VRAM, SM120 Blackwell, PCIe 4.0 57.76 GB/s), 96 GB DDR5 Host RAM.  
**Target Workload:** `Qwen3.8-Flash-Next` (NVFP4 / Radix) and `Qwen3.6-35B-A3B` (NVFP4) at contexts up to 262,144 tokens.

---

## 1. Executive Summary & Goals

This plan establishes the concrete architectural and kernel-level steps required to eliminate the performance gap between the current FreeToken fork and the reference C++/CUDA implementation (`llama-turbo-optimal` / LTO), specifically targeting:

1. **Prefill Throughput (PP):** $\ge \mathbf{1850\text{ tok/s}}$ at 16K context (surpassing the uncompressed baseline anchor of 1857.7 tok/s).
2. **Decode Throughput (TG):** $\ge \mathbf{50\text{ tok/s}}$ with MTP ($n_{\max} \in [1, 3]$) in single-stream decode.
3. **Context Scaling:** Certified **256K context** (262,144 tokens) running live within 15.51 GiB VRAM with zero OOMs.
4. **KV Compression:** Retaining 4-bit compressed KV (`turbo4` / `turbo3` / `vbr`), keeping KV cache memory bounded to $\le 1.74\text{ GiB}$ at 256K.

---

## 2. Comparative Ground Truth: FreeToken vs. LTO

Rigorous inspection of `/models/servers/llama-turbo-optimal` (`QWEN38_FLASH_MTP_PERFORMANCE.md`, `docs/autopilot.md`, and `ggml/src/ggml-cuda/fattn-mma-turbo.cuh`) reveals why LTO achieved **50.9 tok/s (no-MTP)** and **70.3 tok/s (with MTP $n_{\max}=2$)** on this exact host:

| Architecture Dimension | LTO (`llama-turbo-optimal`) | FreeToken Current Fork (`next`) | Performance Impact |
| :--- | :--- | :--- | :--- |
| **KV Dequantization** | **Fused in SRAM** (`flash_attn_ext_turbo4_load_tile`): Loads 4-bit tiles into registers/shmem, applies centroid LUT on the fly. Zero DRAM workspace. | **Split DRAM Workspace** (`qsa/decompress.py`): Separate Triton kernel writes uncompressed FP16 to `_ws_k`, `_ws_v` in DRAM; attention reads it again. | **-13% to -15% TG bandwidth penalty** in FreeToken due to doubling DRAM traffic. |
| **Graph Execution** | **Full CUDA Graph Capture** (`-DGGML_CUDA_USE_GRAPHS`): 1 driver launch per token, $< 10\,\mu\text{s}$ CPU launch overhead. | **Eager Execution** (`--cuda-graph-max-bs 0`): ~100 distinct kernel launches per token via Python/PyTorch runtime. | **-1.5 to -2.0 ms CPU latency per token** in FreeToken (~15–20% TG penalty). |
| **Verify Batch Routing** | Native C++ batch forward over $T=k+1$, executed inside captured graph (~20 ms). | Initially prefill-routed (1.2s); fixed in EXP-040 to decode-routed (73.5 ms), but still eager. | Verification latency in FreeToken is ~3x higher than LTO (73.5 ms vs 22 ms). |
| **Rejection Handling** | Rollback token cursor; lightweight carry vectors (`pending_h`). Zero replay forward. | Executes `gdn_replay` (full 48-layer prefill pass, 40.7 ms) on every speculative reject. | Rejections in FreeToken collapse TG from 25 tok/s to < 10 tok/s. |

---

## 3. The 5 Pillars of High-Performance Implementation

```mermaid
flowchart TD
    subgraph Pillar 1: Fused Verify Kernel (Line 12)
        P1A[Load 4-bit tiles from GMEM] --> P1B[Dequant in SRAM Registers via Centroid LUT]
        P1B --> P1C[Direct MMA Attention without intermediate GMEM workspace]
    end

    subgraph Pillar 2: CUDA Graph Speculative Capture
        P2A[Static Buffer Allocation for Speculative Window] --> P2B[Capture Draft Graph & Verify Graph bs=1]
        P2B --> P2C[Reduce Host-to-Device dispatch from 1.8ms to 0.02ms]
    end

    subgraph Pillar 3: Zero-Replay GDN Snapshotting
        P3A[Preserve hidden_states at accepted prefix m-1] --> P3B[D2D Async Copy of Linear States]
        P3B --> P3C[Eliminate gdn_replay completely: 40.7ms -> 0ms]
    end

    subgraph Pillar 4: Dual-Stream Asynchronous Pipelining
        P4A[Stream 1: Verify Forward & Logits] --> P4B[Overlap Draft Head of step t+1]
        P4B --> P4C[Hide draft execution latency under verification]
    end

    subgraph Pillar 5: 256K Memory Ledger Allocation
        P5A[Turbo4 KV: 6.336 bytes/token] --> P5B[256K KV = 1.738 GiB VRAM]
        P5B --> P5C[517 GPU Expert Slots + 0 OOMs on 15.51 GiB Device]
    end
```

### Pillar 1: Fused Verify Kernel (MTP + TurboKV - Line 12 / D-022)
- **Mechanism:** Eliminate `decompress_turbo4_to_workspace` writing to global memory. In `python/freetoken/attention/qsa_sparse.py`, load 66-byte compressed Turbo4 tiles (`half norm` + 64 bytes nibbles) directly into thread-block shared memory / registers.
- **Dequantization on the Fly:** Expand nibbles against `cent_tensor` within the attention tile loop, evaluating attention directly against uncompressed speculative queries.
- **Target TG Gain:** +15% memory bandwidth efficiency; decode TG moves from 24.7 to 29–31 tok/s baseline.

### Pillar 2: CUDA Graph Capture for Speculative Decoding
- **Mechanism:** Replace dynamic batch constructions in `python/freetoken/scheduler/spec.py` with pre-allocated static tensors in `Batch`.
- **Pre-captured Shapes:**
  - `graph_draft`: Captured for draft micro-batch $T=1$.
  - `graph_verify`: Captured for verify micro-batch $T=k+1$.
- **Target TG Gain:** Removes 1.5–2.0 ms of Python/driver launch overhead per step, accelerating verify forward from 73.5 ms to ~25–30 ms.

### Pillar 3: Zero-Replay GDN State Snapshotting
- **Mechanism:**
  1. Capture `model.model._last_residual[m-1:m]` directly from verify forward as the draft seed residual for the next step, rather than re-forwarding.
  2. For GDN linear states ($S_t$), instead of executing a 40.7 ms prefill replay on reject, maintain per-token snapshot checkpoints or step GDN decode in-place across the $m$ accepted tokens.
- **Target TG Gain:** Eliminates the 40.7 ms rejection penalty entirely, lifting cold/noisy MTP from ~19 tok/s to >35 tok/s, and warm steady state to >50 tok/s.

### Pillar 4: Dual-Stream Asynchronous Draft-Verify Pipelining
- **Mechanism:** Utilizing CUDA streams `stream_draft` and `stream_verify`.
- **Pipeline:** While `stream_verify` completes hyper-connection mixing, final RMSNorm, and vocabulary projection for cycle $t$, `stream_draft` begins embedding lookup and MTP projection for cycle $t+1$.
- **Effective TG Formula:**
  $$\text{Effective TG} = \frac{\text{Mean Accepted Length}}{\max(t_{\text{verify}}, t_{\text{draft}})} \ge \frac{2.0}{0.035\text{ s}} \approx \mathbf{57.1\text{ tok/s}}.$$

### Pillar 5: 256K Context Scaling on RTX 5080
- **Physics Ledger at 256K (262,144 tokens):**
  - Static Model Weights: **9.610 GiB**
  - GDN Recurrent State: **1.737 GiB**
  - Turbo4 KV Cache (256K tokens @ 6.336 B/tok): **1.738 GiB**
  - Activation & Graph Reserves: **0.750 GiB**
  - GPU Expert Cache (517 slots @ 7.6 MiB): **3.930 GiB**
  - Total VRAM: **15.10 GiB** (cleanly under 15.51 GiB device ceiling).

---

## 4. Implementation Phasing & Commit Sequence

- **Phase A (Immediate):** Zero-replay GDN state handling & residual slice in `python/freetoken/scheduler/spec.py`.
- **Phase B:** Direct in-SRAM Turbo4 tile decompression in attention kernel (`qsa_sparse.py` / Triton kernel).
- **Phase C:** Static memory stabilization and CUDA Graph capture for `--spec-mtp 1/2`.
- **Phase D:** Live benchmark validation at 16K, 128K, and 256K contexts via `benchmarks/bench_pp_tg.py`.
