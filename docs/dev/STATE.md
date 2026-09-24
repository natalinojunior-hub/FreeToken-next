# State — 2026-09-24 (campaign 13 in progress; campaign 12/11 state at `git show 0e9d00c:docs/dev/STATE.md`)

Anchor: UD-IQ4_XS cold graph-on k0 4096/256: PP 1467, TG 47.28, hash `3af3056b98c0` (re-verified after `b5d4243`).
Done (`b5d4243`): GGUF tokenizer control/user tokens atomic (chat prompts were corrupted: <think>/<tool_call> split); ggml Q2_0; tiled ssm_out input gather for 256-block types; split MTP head load; `find_gguf_tensor`.
Done: ISTA GSQ-RCO IQ3_XXS loads: 4K TG 59.06 (+24.9%) PP 1651; 16K TG 58.49 (+28.4%) PP 2360; slots 5555; usage eval 20/20.
Done: KV matrix — turbo3/turbo4 cost 11-17% TG at 4K/16K; capacity lever only. KV-in-RAM deferred until engine final.
Finding: 16K prefill transient (chunk 8192, 2.02 GiB) costs ~550 expert slots; static chunk cap gives +3.4% TG but -34% PP (rejected).
Decision: k0 stays default; k1 opt-in. No tests >16K until engine final. Engine stays universal (no model hardcoding).
Decision: MTP standalone Q8 head = same bytes as shared-Q8_0 (no perf gain from fc_hidden tensor).
Evidence: `/models/desenvolvimento/ft-campaign2/campaign13/LEDGER.md`.
Done: usage eval post tokenizer fix: UD 20/20 (was 12/20), AD 20/20, ISTA 20/20.
Done: MTP k1 16K: ISTA sq8 TG 63.65 (+8.8%, accept 86.8%), UD sq8 49.27 (+8.2%, 84.8%); 4K accept ~59-61%, Q4 heads best (+1.8%); k2 4K -28%.
Next: `/models/desenvolvimento/ft-campaign2/campaign13/CODEX-CONTINUE.md` (MTP acceptance >75% at k1/k2/k3 per head/model, pfeifferj split shards, prefill borrows expert arena).
