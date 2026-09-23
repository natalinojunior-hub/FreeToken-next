# State — 2026-09-23

## Doing
- Nothing in progress. The k=1 spec-verify CUDA graph is committed.

## Done
- 28f7ec6: the disk PLE now stages the draft token's row in the verify window (it read a stale row before).
- Verify graph (k=1): the whole 2-token verify forward is one CUDA graph, with the eager path kept as fallback.
  - 16K turbo3 k1: verify 54.0 -> 37.1 ms, TG 27.85 -> 38.12. Draft, replay, acceptance and sha unchanged.
  - Launches per verify: 5975 -> 3. GPU idle: 31.5 -> 1.9 ms (nsys). 17 ms of the rest is expert H2D copies.
  - `FREETOKEN_VERIFY_GRAPH_CHECK=1` checks every verify bitwise, eager against graph. `FREETOKEN_VERIFY_GRAPH=0` forces eager.
- make ci PASS.

## Decisions
- The graph emits no GDN checkpoints. spec.py hard-codes `checkpoints = None`, so none is ever restored.

## Next
1. Zero-replay for GDN+QSA+PLE (replay 16 ms on 27% of cycles). Checkpoints need static buffers.
2. Expert miss copies (17 ms per verify).
