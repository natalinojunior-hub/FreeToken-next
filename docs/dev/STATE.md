# State Snapshot — 2026-09-20

## Doing Now
- MTP speculative decode optimization for Qwen3.8-Flash-Next-NVFP4-Radix
- Achieved: SHA1 bit-identical equivalence (k=0 vs k=4), PP ~1732 tok/s, TG ~27.8 tok/s at K=4

## Done This Session
1. Fixed MTP warmup: `warmup_mtp_draft_kv` now populates draft head KV over full prefill context (removed `spec_logits_indices`)
2. Implemented `adaptive_mtp.py` with confidence-gated draft launch (top-1 prob threshold)
3. Fixed `_last_residual` update after commit to seed next draft chain from last accepted token
4. Identified optimal K=4: 27.8 tok/s TG (vs baseline 30.0 tok/s, -7% regression), PP maintained at 1732 tok/s
5. Verified SHA1 equivalence: k=0 and k=4 both produce `573a19610680`
6. All tests pass (`make ci`: 1953 passed, 206 skipped)

## Blocked / Limitations
- TG still -7% below k=0 baseline due to token-a-token replay in decode path
- Need batched multi-token decode replay kernel for +35% TG target (39.7 tok/s)
- Acceptance rate ~40-50% even with warmup fix (draft head accuracy ceiling)
- GGUF Unsloth-IQ4_XS MTP crashes (missing MMQ dequant kernel for decode)

## Next Steps
1. Implement batched multi-token decode replay (decode path with extend_len > 1)
2. Add draft head confidence thresholding to skip low-probability drafts
3. Test K=4 with adaptive confidence gating enabled
4. Benchmark LTO engine (llama-server) for comparison baseline

## Metrics Anchors (Qwen3.8-Flash-Next-NVFP4-Radix, 4K context, naive cache)
| Config | PP (tok/s) | TG (tok/s) | SHA1 |
|--------|-----------|-----------|------|
| k=0 (baseline) | 1768 | 30.0 | 573a19610680 |
| k=1 | 1740 | 24.5 | 573a19610680 |
| k=2 | 1740 | 24.3 | 573a19610680 |
| k=3 | 1739 | 18.8 | 573a19610680 |
| **k=4 (optimal)** | **1732** | **27.8** | **573a19610680** |
| k=5 | 1733 | 14.8 | divergent |
| k=6 | 1733 | 13.2 | divergent |

## Files Changed
- `python/freetoken/scheduler/spec.py`: warmup fix, _last_residual update, confidence gating
- `python/freetoken/scheduler/adaptive_mtp.py`: new module with AdaptiveMTPController
- `tests/scheduler/test_adaptive_mtp.py`: legacy API compatibility maintained

## Commands to Reproduce
```bash
# Baseline k=0
FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1 python benchmarks/bench_pp_tg.py --model /models/Qwen3.8-Flash-Next-NVFP4-Radix --tokens 4096 --decode 64 --repeats 3 --label k0_baseline --mem-ratio 0.98 --serve-arg '--cache-type naive' --serve-arg '--max-running-requests 1'

# Optimal MTP k=4
FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1 python benchmarks/bench_pp_tg.py --model /models/Qwen3.8-Flash-Next-NVFP4-Radix --tokens 4096 --decode 64 --repeats 3 --label k4_optimal --mem-ratio 0.98 --serve-arg '--spec-mtp 4' --serve-arg '--cache-type naive' --serve-arg '--max-running-requests 1'
```

