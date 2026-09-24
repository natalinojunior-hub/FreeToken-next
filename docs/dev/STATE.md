# State — 2026-09-24 (campaign 10 closed; campaign 9 state at `git show ee0283c:docs/dev/STATE.md`)

Done: Gate A — commit `9a640ef` removed dead unsafe `_adaptive_mtp_controller` branch in `spec.py`, landed opt-in token trace + bench `--token-trace`/`--fresh-server-each-repeat`, added `freetoken/debug/__init__.py`. `make ci` 2111 passed.
Done: cold graph-on k0 4096/256 control: PP 1466.7, TG 47.28, VRAM 14.70 GiB, RSS 80.31 GiB, hash `3af3056b98c0`, slots 3519.
Done: nsys attribution (TG 46.52, −1.6% overhead): per step 20.24 ms = gather 7.57 (PCIe ceiling ≈52 GB/s) + dense q8_0 5.16 (≈745 GB/s) + routed GEMV 1.79 + cuBLAS 1.23 + host gap 0.84 + rest.
Decision: Gate C NO-GO — shared-expert/gather overlap ≈1.7% ceiling, host gap 4.1% max, others already no-go. No candidate, no A/B.
Decision: decode is bus-bound (PCIe for missed experts, VRAM for dense q8_0). k0 stays default.
Evidence: `/models/desenvolvimento/ft-campaign2/campaign10/ATTRIBUTION.md`, `nsys_decode.py`, `k0-nsys.sqlite`, PERFORMANCE.md last entry.
Next: +5% needs fewer bytes per token — smaller expert/dense formats (needs authorized conversion) or better hit rate (no deployable policy known). No open engineering lever.
