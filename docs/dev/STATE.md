# State — 2026-09-23

Doing: nothing in progress. Campaign 2 closed; report /models/desenvolvimento/ft-campaign2/RELATORIO.md.
Done: GGUF chosen over native NVFP4 (TG 2x). Offload default (hybrid -8% with benched split). Deterministic cold prefill (211efb6). cache_prompt:false honored. Greedy at temperature 0 regardless of top_p (MTP now runs for such requests). `ft tune` profiles applied at boot: 16K turbo3 picks MTP k1 (43.88 vs 40.48).
Decisions: docs/dev/DECISIONS.md D-024..D-028.
Next:
1. Draft-step CUDA graph: compute-sanitizer memcheck on replay (illegal access in qsa_forward), ~2 ms/cycle.
2. Verify-window hit/miss overlap (miss gather serialized with GEMV).
3. turbo3 verify-vs-decode KL outliers (0.97 at one position) - open, bf16 explained by cuBLAS M=1 vs M>=2.
4. >100 tok/s not met (cold 40-44); 128K/256K runs gated on it.
