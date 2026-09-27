docs/dev/STATE.md and LESSONS.md missing -> parent dir docs/dev/ was deleted while symlinks at repo root remained -> recreate dir before writing through the symlink
ft-campaign2 scripts/gpu.lock path missing -> dir moved to old/freetoken-next/external/ft-campaign2 -> use the archive path for campaign dirs and gpu.lock
256K bench failed 'corpus slice gave 238188' -> default corpus too small and rope table caps prompt+decode at 262144 -> use campaign26/prompt-470k.txt and --tokens 261824
expert-cache rebuild took 1.2 s -> gc.collect() scanned the 76 GiB-RSS startup heap -> gc.freeze() after scheduler startup
VRAM guard shrank plan 15% after VMM -> VMM memory is outside torch allocator, peak window straddled a resize -> reset peak stats at every in-place resize
illegal memory access in prefill with VMM -> bank_views chose staging by shape caps, read unbacked rows -> decide staging by live/planned caps
