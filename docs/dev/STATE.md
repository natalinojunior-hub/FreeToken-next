# State — 2026-09-23

## Doing
- Nothing in progress. Overnight GGUF campaign committed (a0adc18..7eb12b5).

## Done
- 4K k1 TG 37.95 -> 44.69 (batched mmvq, host token embedding, deferred replay). k0 unchanged.
- Cold PP 377 -> 1472 (4K), 398 -> 2064 (15.7K): dequant+cuBLAS dense, dequant+bf16 fused-MoE for >= 512 tokens.
- Old PP anchors were cached-prefix PP. Cold PP needs a unique prompt per request (ft-campaign/cold.sh).
- `make bench-flash MTP=1` runs the Flash-Next GGUF bench. make ci PASS (2003).
- Details and matrix: docs/dev/PERFORMANCE.md "Overnight campaign"; logs in /models/desenvolvimento/ft-campaign/.

## Decisions
- Deferred replay: up to 2 rejected-window tokens ride in the next verify (graphs for 2-4 rows). FREETOKEN_SPEC_DEFER_REPLAY=0 disables.
- >100 TG not reachable on this path: ~16 ms/verify of PCIe expert copies. No >16K runs (gate unmet).

## Next
1. MTP draft CUDA graph: illegal access on replay (patch ft-campaign/draft-graph.patch), ~2 ms/cycle.
2. Dequant MoE prefill floor ~24 ms/layer: tune fused_moe config, dequant straight into the kernel.
3. Expert copy overlap with hit/shared-expert compute inside the verify graph.
