# PROVENANCE — freetoken-next

Authoritative record of what this tree is, where every imported mechanism came from,
and the toolchain every measurement below was produced with.

## 1. Source lineage

| Item | Value |
|---|---|
| Target repo | `/models/desenvolvimento/freetoken-next` |
| Base | official FreeToken, `https://github.com/FlashML-org/FreeToken` (remote `upstream`) |
| Upstream base SHA | `cac247a860e316e06580d05aeb05f2e647bde214` |
| Tag at base | `v0.1.3` (2026-09-15 16:47:35 -0700, `chore(release): 0.1.3 (#489)`) |
| `git fetch upstream` on 2026-09-16 | `HEAD == upstream/main` → base is current tip, 0 commits behind |
| Working branch | `next` (branched from `main` = `upstream/main` at the SHA above) |
| Package version | `python/freetoken/version.py` → `0.1.3` |
| License | Apache-2.0 (`LICENSE`, unchanged) |
| Tracked files at base | 626 (`git ls-tree -r HEAD`) |

History note: the repo was cloned with an empty working tree; `git checkout -f next`
restored it. No upstream file was modified by that operation (verified: `git status`
clean apart from `.qwen/`).

### Reference installs on this host (not the base of this tree)

- `/models/servers/freetoken/venv` — installed wheel `freetoken 0.1.2+gaf71ba432`
  (build stamp commit `af71ba432` = `ci: publish engine-<platform>.json manifests ... (#377)`),
  plus `freetoken_kernel_cache 0.1.2+cu130.gaf71ba432`. **This is the build the measured
  anchors in PERFORMANCE.md were produced with** — it is 23 commits behind `v0.1.3`.
- `/models/servers/llama-turbo-optimal` — llama.cpp-derived fork ("LTO"); reference
  implementation only (TurboQuant/TCQ/VBR, qwen4exp MTP). Not a runtime dependency.

## 2. Host toolchain (measured 2026-09-16)

| Component | Value |
|---|---|
| OS | Linux (Ubuntu), host `rtx5080` |
| CPU | AMD Ryzen 9 9900X, 12 physical cores / 24 threads (`nproc` = 24) |
| RAM | 91 GiB total (`free -g`) |
| GPU | NVIDIA GeForce RTX 5080, 16303 MiB, compute capability **12.0 (SM120)** |
| GPU UUID | `GPU-85428346-1503-a189-0b52-7eb8049199f1` |
| Driver | 610.57.04 |
| System NVCC | CUDA 13.3, V13.3.73 (`/models/outros/cuda-13.3`) |
| gcc | Ubuntu 12.5.0 |
| cmake | 4.4.2 |
| System python | 3.14.4 (**not** used; torch has no 3.14 build here) |
| Env python | 3.12.14 (uv-managed CPython at `~/.local/share/uv/python/cpython-3.12.14-...`) |
| torch (reference env) | 2.11.0+cu130 → `libcudart.so.13`; NVCC major 13 matches, so `kernel/_toolchain.py::check_nvcc_matches_torch` passes |
| triton (reference env) | 3.6.0 (pinned `triton==3.6.0` by `pyproject.toml`) |
| uv | 0.12.10 at `/models/servers/freetoken/bin/uv` |
| `UV_CACHE_DIR` | default `/models/outros/cache/uv-cache` is **root-owned → unwritable**; this tree uses `/models/desenvolvimento/.uvcache` |
| freetoken-next env | `.venv/` in this repo, editable install of `[accel]` extra |

`pyproject.toml` dependency contract: `torch>=2.11,<2.12` (PyPI wheel is the cu130 build),
`triton==3.6.0`, `flashlib==0.3.0` (device-side LRU admission kernel for the MoE expert
cache), `apache-tvm-ffi==0.1.13.post3`, `gguf>=0.19,<1` (already a core dependency),
`transformers>=5.16,<5.17`, `numpy>=2.0,<2.5`; extras `fi` (flashinfer-python[cu13]) and
`sgl` (sglang-kernel==0.4.5) make up `[accel]`. Native extensions built at install time
by `setup.py`: `freetoken.kernel._pinned_tensor`, `freetoken.kernel._cpu_moe`,
`freetoken.kernel._ple_store` (all CppExtension, CUDA-header/dirs only, `-O3 -std=c++17`).

## 3. Local model corpus (inputs for phases 1–2)

| Path | Format | Notes |
|---|---|---|
| `/models/Qwen3.6-35B-A3B-NVFP4-FT` | HF safetensors, 22 GB | arch `Qwen3_5MoeForConditionalGeneration` / `model_type=qwen3_5_moe`; ModelOpt `hf_quant_config.json`: `quant_algo=MIXED_PRECISION`, **`kv_cache_quant_algo=FP8`**, experts `W4A16_NVFP4` group_size 16, linear_attn projections FP8 |
| `/models/Qwen3.8-Flash-Next-NVFP4-Radix` | HF-style dir with per-layer expert shards (`layer-XXXXX-experts-*.safetensors`) | qwen4_exp / Qwen3.8-Flash-Next NVFP4, RadixArk repack |
| `/models/Qwen3.8-Flash-Next-Unsloth-IQ4_XS/{MTP,UD-IQ4_XS}` | GGUF (+`mmproj-F16/BF16.gguf`) | Unsloth quant incl. an `MTP` dir |
| `/models/Qwen3.8-Flash-Next-AD-4.27/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64` | GGUF dir | Q4_K_M |
| `/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-MTP-Q4XS-Q3S.gguf` | GGUF | dense 27B, IQ3_S + MTP heads |
| `/models/Ornith-1.5-35B-A3B-APEX-MTP-I-Compact.gguf` | GGUF | MoE A3B + MTP |
| `/models/Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf` | GGUF | MoE A3B + MTP |
| `/models/qwen36-35b-a3b-dflash-Q4_K_M.gguf`, `/models/dflash-draft-Ornith15.gguf` | GGUF | draft models |
| `/models/servers/prompt-{25600,26k,235k}.txt` | text | long-prompt corpora for context benchmarks |

Details (tensor families, quant types, MTP keys per file) are recorded by the GGUF/MTP
audits in ARCHITECTURE.md and EXPERIMENTS.md.

## 4. Imported / adapted code

**Status: none yet.** No code has been copied out of `llama-turbo-optimal`, FreeToken-Kai,
or any fork into this tree as of this writing. Every future import gets a row here with:
origin repo URL, commit SHA, PR/issue if any, files touched, the local modification
summary, and the license of the origin.

Upstream FreeToken commits that this tree already contains (part of the `v0.1.3` base,
listed here because later phases build on them):

| SHA | Subject | Relevance |
|---|---|---|
| `477c860` | refactor(quant): config, scheme and method layers for quantization (#418) | the QuantConfig/QuantScheme/QuantMethod abstraction new KV backends must plug into |
| `fb7f732` | refactor(quant): hand the checkpoint QuantConfig to the weight readers (#427) | checkpoint→reader quant handoff |
| `0ffd5c8` | feat(qwen3_5_moe): read every checkpoint layout through the QuantConfig (#438) | Qwen3.5/3.6 MoE read path |
| `fa814ab`, `ddd2e3a` | fix/feat(qwen4_exp): expert quant kind from QuantConfig; block-fp8 dense projections natively (#426/#428) | Qwen3.8-Flash-Next read path |
| `46d2743` | fix(scheduler): reserve paged KV at allocation granularity (#367) | KV/VRAM accounting behaviour to preserve |
| `3d919e9` | fix(checkpoint): write the Qwen3.8-Flash-Next PLE table next to the FTW (#420) | PLE storage |
| `e0886cc` | fix(models): compute shared experts before in-place routed experts (#463) | MoE correctness |

## 5. External prior art identified on 2026-09-16 (GitHub API, unauthenticated)

Nothing here is copied into this tree yet; this is the audit map the phases must consult
before writing an equivalent mechanism. Upstream `FlashML-org/FreeToken`: 12 974 stars,
`main` @ `cac247a`, Apache-2.0, last push 2026-09-16.

### 5.1 Official roadmap issue #79 "FreeToken Roadmap (2026)"

Landed: Qwen3.8-Flash-Next support incl. **PLE pinned in RAM (#257)** and **PLE offloaded to
disk (#311)**; quant layer refactor (#418, #427); image input for Qwen/Gemma-4/GLM-5.3/
Muse-Glimmer/MiniMax-M3. **Still open — i.e. our lane**: GGUF checkpoints and their quant
types across architectures; speculative decoding (MTP / DFlash / DSpark); DeepSeek-V4.1;
TP; ROCm; Apple Silicon.

### 5.2 Open PRs fetched as local refs (`git fetch upstream pull/N/head:refs/pr/N`)

| Ref | PR | Content | Net delta vs merge-base |
|---|---|---|---|
| `refs/pr/354` | #354 | KV as fp8 e4m3 codes behind `--kv-cache-dtype {auto,bf16,fp8}`; uint8 code buffer with the *same geometry* as the 16-bit KV buffer + one fp32 row scale; `BackendInfo.supports_fp8_kv` gating; claims it "is what lets Qwen3.8-Flash-Next serve a 1M-token context on this card" | 34 files, +3186/−189 (head `9b103b0`, 2026-09-08) |
| `refs/pr/408` | #408 | `--kv-cache-dtype nvfp4`: packed E2M1 + one E4M3 scale per 16 values + one FP32 row scale per token/KV head; **76 B vs 256 B per K/V row at head_dim 128**; pool budgeting/rebuild, CUDA-Graph-safe KV writes, Triton restore paths, microbenchmark; states the restore path is currently *slower* than BF16/FP8 | 41 files, +4659/−224 (head `1c7bb42`, 2026-09-14) |
| `refs/pr/113` | #113 | 8-bit DSV4 window/compressed KV behind `--kv-cache-dtype` (closed) | 12 files, +577 |
| `refs/pr/460` | #460 | DeepSeek-V4.1 with native fp8-fp4 KV storage | 132 files, +14829 |
| `refs/pr/69` | #69 | [2/3] exact DSpark **speculative decoding** for DeepSeek-V4 (the only spec-decode machinery upstream) | 59 files, +5752 |
| `refs/pr/337` | #337 | **NVMe disk tier for MoE expert banks** | 14 files, +1475 |
| `refs/pr/447` | #447 | owner-local expert parallelism + TP | 49 files, +5030 |
| `refs/pr/494` | #494 | fix(gemma4): load **mixed-quant GGUF** checkpoints → GGUF reading exists on the base, narrowly | 13 files, +575/−54 |

### 5.3 Issues that are direct evidence for our phases

- **#141** (open, 2026-08-24) "Add support for all major 2026 KV cache compression
  techniques (**TurboQuant**, RotorQuant, KIVI, KVQuant, eviction)": TurboQuant described as
  *PolarQuant rotation + QJL residual, 3–4 bit, 5–6× compression*; reporter asks for a unified
  KV backend that is LRU-expert-cache compatible. **Not implemented upstream** → the
  Turbo3/Turbo4 backend is our differentiator, and this issue is the demand evidence.
- **#421** (open, 2026-09-09) "MTP for qwen4_exp … planned beyond DSV4?": states that
  `models/qwen4_exp/weight.py` **drops `mtp.*` outright**, that the shipped head is
  `mtp.layers.0.mlp.experts.*` (128×128 block-FP8, ~3.1k tensors), and that on an RTX 3090
  decode is ~19 tok/s vs ~910 tok/s prefill on a 39k prompt — i.e. expert transfer per decode
  step is the exposed cost that verifying γ tokens in one pass would amortise. Confirms our
  Phase 9 target and gives the tensor-name starting point.
- **#409** (open) Flash-Next NVFP4 cannot start on 12 GB: **~9.9 GiB of unquantized BF16**
  resident — a VRAM-ledger/governor symptom (Phase 6 evidence).
- **#436** (open) hybrid MoE 20–40× slower than offload on an offload-heavy box.
- **#396** (open) feature request: int4 routed experts AWQ→Q4_1, pack-quantized→Q4_0.
- **#407** (closed) GGUF of Qwen3.6-35B-A3B fails: `qwen35moe architecture not supported`.
- **#200** (open) SWA prefix cache loses cross-request reuse when a request diverges.

### 5.4 FreeToken-Kai (the fork named by the mission)

`yuuki-net/FreeToken-Kai`, cloned for audit at
**`/models/desenvolvimento/reference/freetoken-kai`** (branch `kai`, HEAD `9dc5412`,
pushed 2026-09-16). Its `main` is `af71ba43` (= the anchor build commit) and `kai` has
already **merged our base `cac247a`**, so `git diff cac247a..HEAD` is exactly its delta:
**191 commits, 203 files, +31 851/−583**. It ships its own docs for the mechanisms we need:
`docs/kai.md`, `gguf.md`, `kv-cache-quant.md`, `bank-ram.md`, `vram-and-speed.md`,
`prefill-chunk.md`, `prefix-reuse.md`, `pipeline.md`, `image-input.md`, `turing.md`.
Observed features (from its commit log): host-RAM MoE **bank file as the only copy of the
experts** (`--moe-bank-ram auto`, `--moe-bank-prefetch`, `--moe-bank-readahead`,
`--moe-bank-rewarm`, `ft bank`, `ft doctor disk`), buffered-vs-O_DIRECT prefill reads,
`--prefill-profile` / `--prefill-chunk-budget`, pipeline parallelism (`--pp-size`,
`--pp-prefill-group`, `--pp-send-ahead`), `--dense-quant fp8`, CPU vision tower
(`--mm-encoder-weights cpu`), `--prefix-disk-cache`, arithmetic e2m1 decode in the prefill MoE
kernel up to Ampere. Target hardware: RTX 3060 12 GB / RTX 2060 / WSL2, claiming a 35B MoE at
250K context. Audit result recorded in ARCHITECTURE.md §7 and per-port provenance rows in §4.

## 6. Contribution policy note

Upstream `AGENTS.md` forbids autonomous agents from pushing/contributing to FreeToken.
This tree is a **local research fork**: no `git push`, no `gh pr create`, no issue
creation. Local commits on `next` only.
