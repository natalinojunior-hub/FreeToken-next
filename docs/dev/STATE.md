# State — 2026-09-23

## Doing
- Nothing in progress. The geometry-pool expert cache is committed.

## Done
- a9d30f4: the MTP draft uses its own blk.48 expert bank (acceptance 55.5% -> 75.2%).
- Geometry pools: one byte budget for all experts, each geometry gets its own LRU range.
  - Resident rows: target 1181 -> ~2650 at k0, and 738 -> 2658 at k1. The MTP bank gets 80 rows.
  - Miss rate at k1: target 83.3% -> 37.8%, MTP 99% -> 49%.
  - Timings at k1: verify 70.2 -> 53.0 ms, replay 20.1 -> 15.4 ms.
  - Speed: k1 TG 22.34 -> 28.42, k0 TG 35.11 -> 41.35. PP unchanged. sha identical before and after.
- make ci PASS.

## Decisions
- Layers are grouped by row geometry. Each layer gets an equal row share, clamped to [decode floor, layers*E].
- A pool below E rows stages its prefill layer at the front of each arena.

## Next
1. Verify step in a CUDA graph (verify 53 ms vs decode ~24 ms).
2. Zero-replay for GDN+QSA+PLE.
