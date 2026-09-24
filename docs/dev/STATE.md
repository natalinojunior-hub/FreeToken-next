# State — 2026-09-24 (campaign 11 closed NO-GO; campaign 10 state at `git show 3f3ba92:docs/dev/STATE.md`)

Anchor: cold graph-on k0 4096/256 UD-IQ4_XS: PP 1464.3, TG 47.24, hash `3af3056b98c0` (paired control, matches campaign 10), slots 3519.
Done: Gate 1 byte model (no source change): LRU replay + measured dense MMVQ time vs Q8_0 — Q6_K 0.942, Q5_K 0.706, Q4_K 0.650.
Ceilings: resident-expert reclaim +1.2..2.7% (reject); dense Q6_K +3.6% (reject); AD-4.27 experts +9.5..10.5%; dense Q5_K +11.5% (conversion not authorized).
Done: Gate 3 AD-4.27 A/B (operator-authorized, order AD/UD/AD): TG 49.68/49.54 vs 47.24 = +4.97% mean, PP +8.5%, slots 4135, VRAM 14.82 GiB, RSS 79.0 GiB, hash `03a0b9bc7715` stable, 0 Tracebacks.
Decision: NO-GO — gain not reproducibly >=5%; realized about half of the modeled ceiling; unlisted checkpoint with unmeasured IQ2_S quality. Quality gate and 15.7K skipped.
Decision: k0 on UD-IQ4_XS stays default. No source change, `make ci` not required.
Evidence: `/models/desenvolvimento/ft-campaign2/campaign11/GATE1.md` (sha256 `0d3e5f968cb2`), `ad-k0.json` `3d6db8592e8d`, `ud-k0.json` `709ddb30fcf9`, `ad-k0-r2.json` `dc8a82f539d5`.
Lesson: byte-model ceilings from the k1 proxy trace overstate realized gain about 2x; discount before authorizing conversions.
Next: only unexplored lever is dense Q5_K conversion (needs new qwen4exp converter + quality gate; expect well below its +11.5% ceiling). >100 tok/s: no measured path.
