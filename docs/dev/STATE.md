doing: stopped at operator request (weekly limit); all work UNCOMMITTED
done: Campaign 26 in-place decode-phase expert residency (CUDA VMM): TG 59.25 cold / 59.77 warm (was 55.20/55.48), PP 2510, SHA 76a5508fd576
done: 64K/128K/256K no OOM; make ci green (2295)
done: audits without gain: KV rebalance cadence, pool caps optimum, fast_index_copy launch params
decisions: decode residency only with VMM (mid-request rebuild changed 64K output)
evidence: /models/desenvolvimento/old/freetoken-next/external/ft-campaign2/campaign26/
next step: expert gather kernel efficiency (42% of decode kernel time, ~30 GB/s vs 53.5 GB/s); see ai-memory notes/campaign26-adaptive-vram-residency
