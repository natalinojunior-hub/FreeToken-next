# State Snapshot — 2026-09-22

## Doing Now
- MTP task (16K ctx, 4096 prompt, 256 decode, greedy). Harness in session scratchpad: tgclient.py/ft.sh/lto.sh. VRAM planner untouched.
- MTP != k=0 text is NUMERIC (forced-reject: verify row0 vs decode replay, rel diff 0.5% at layer 0 growing to ~5%; 2/255 argmax flips), not a state bug. LTO has the same flip. make ci PASS (1972 passed).
- MTP still < k=0: verify 59 ms (eager, 2 tokens) vs decode 28 ms. Even with 85% acceptance k=1 ~= 34.6 TG. Needs: own MTP expert bank + graphed verify + zero-replay (PLE/QSA tape).

## Done (uncommitted)
1. LTO same GGUF, turbo3: k=0 40.18; MTP k1 41.08 k2 41.97 k3 43.06 k4 43.70 (best, +8.8%) k5 39.68 k6 32.82. LTO greedy non-deterministic; its k>=2 uncertified.
2. GGUF fixes (models/qwen4_exp/gguf.py, layers/gguf.py): LM head ignored spec_logits_indices (0% acceptance); plus-one norms loaded +1 (llama.cpp folds); GDN V heads tiled not undone; output_gate silu->sigmoid. GGUF was producing gibberish; now "Berlin". GGUF k0 35.34 TG (was 28.95 garbage). Test: tests/models/qwen4_exp/test_gguf_layout.py.
3. FT GGUF MTP after fixes: k1 22.47 (55% accept), k2 15.4, k3 11.6, k4 10.1. Verify 59 ms, replay 16 ms, draft 3 ms vs decode 28 ms.

## Next
1. Fix MTP parity. 2. MTP MoE aliases target layer-0 experts (wrong; own blk.48 exps exist) -> low acceptance vs LTO 85%. 3. Drop unused per-step GDN clones; graph verify. 4. make ci. Commit only when operator asks.
