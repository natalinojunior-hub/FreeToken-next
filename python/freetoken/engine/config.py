from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from freetoken.distributed import DistributedInfo
from freetoken.layers.quantization import set_quant_config
from freetoken.mm.config import ENCODER_SECTIONS, MultimodalConfig
from freetoken.models.register import (
    EncoderSpec,
    ModelSpec,
    _load_attr,
    checkpoint_quant_config,
    get_model_spec,
)
from freetoken.utils import cached_load_hf_config, init_logger

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

logger = init_logger(__name__)


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 4
    attention_backend: str = "auto"
    # KV slab format: auto/bf16 is the paged bf16 cache; turbo3/turbo4 store the full-attention
    # group as rotated 3/4-bit codes, so the KV a long context buys out of the expert cache is
    # ~4x smaller. Only the triton backend can read coded tiles.
    kv_format: str = "auto"
    # Set by resolution when kv_format was "auto": the engine may then step the KV format down
    # (engine._KV_FIT_LADDER) until the requested context fits, before refusing it.
    kv_format_auto: bool = False
    moe_strategy: str = "auto"
    # old name of moe_strategy; __post_init__ folds it in
    moe_backend: str | None = field(default=None, repr=False)
    # --quant-backend: layer[.kind]=kernel entries, comma separated
    quant_backend: str | None = None
    # PLE table backend: "disk" (default) reads rows from the checkpoint files per fill, "pinned" preloads the table into page-locked host RAM.
    ple_backend: str = "disk"
    # Expert-bank host load (--expert-load): auto|serial|parallel. "auto" reads scattered
    # experts in parallel but falls back to serial when free RAM can't cover the banks + the
    # parallel reader's extra (non-reclaimable) whole-shard buffer; "serial" forces the
    # low-memory reclaimable read; "parallel" forces the fast read.
    expert_load: str = "auto"
    moe_cache_size: int = 0
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    kv_reserve_tokens: int = 8192  # KV floor for --moe-cache-auto; small by design (MoE-priority)
    # auto: KV in RAM whenever the pool supports it and RAM holds it (measured faster from 64K);
    # otherwise all-VRAM.
    kv_tiering: str = "auto"
    # RAM-tier size in tokens (--kv-ram-tokens), consumed only by kv_tiering="force"
    # (rounded up to whole pages by the engine); 0 gives every context token a RAM page.
    # Ignored (no effect) when kv_tiering="off".
    kv_ram_tokens: int = 0
    # RAM-tier element type: auto = the KV dtype when it fits the safe RAM budget, else FP8.
    kv_ram_dtype: str = "auto"
    # Buy the serving context out of the expert cache instead of hand-tuning the floor above:
    # the plan funds max_seq_len of KV and sizes experts from what remains, and refuses with the
    # shortfall when that context is not affordable. Opt-in, because it trades decode speed for
    # reach, and the trade is the operator's call until the plan can price it (see D-015).
    kv_reserve_context: bool = False
    allow_rope_extend: bool = (
        False  # auto-extend RoPE table past checkpoint max_position for 512K/1M
    )
    moe_cache_policy: str = "lru"
    # Mixed-geometry banks only: comma-separated per-pool slot caps, ordered as
    # cache_budget.expert_pools sorts them (largest layer group first). "" = uniform
    # layer-count split. Experimental knob; validated against the pools at init.
    moe_pool_caps: str = ""
    moe_prefill_overlap: bool = True
    # Prefill hit/miss split: serve cache-resident experts D2D during prefill
    # prefetch instead of re-streaming the full layer over PCIe. Needs CUDA >= 12.8
    # (cudaMemcpyBatchAsync); no-op unless moe_cache_size > 2 * num_experts.
    moe_prefill_hit_d2d: bool = False
    moe_collect_stats: bool = False  # capture decode miss-rate counters into the cuda graph
    # CPU MoE backend (--moe-strategy cpu): number of CPU worker threads computing
    # the decode experts. 0 = auto (physical cores). Ignored by other backends.
    moe_cpu_threads: int = 0
    # Hybrid CPU/GPU decode (--moe-strategy offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-strategy cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # Hybrid MoE backend (--moe-strategy hybrid): max experts fetched over PCIe per
    # (layer, decode step); the rest of that step's misses are computed on the CPU.
    # -1 (default) = auto: fetch the benched pcie_bw/cpu_bw fraction of each step's
    # misses so the PCIe fetch and the CPU compute finish together (perfect overlap);
    # falls back to a fixed cap of 1 without a usable `ft bench bw` profile.
    moe_hybrid_max_fetch: int = -1
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    # Explicit cap only: the ceiling is min(ratio x baseline, baseline - modelled reserve), and
    # the modelled reserve (vram_ledger) is what protects the runtime peaks -- a 0.9 default
    # cost ~300 expert slots on Tiel 16K for nothing (campaign 19).
    memory_ratio: float = 1.0
    # Hybrid GDN models default to the HybridRadixCache (cross-request GDN-state prefix reuse);
    # `--cache-type naive` opts out. linear_state_cache_ratio sizes the GDN snapshot cache as
    # ceil(ratio * max_running_req) extra slots.
    linear_state_cache_ratio: float = 2.0
    # Window/full ratio for the SWA radix cache (`--cache-type radix` on SWA models) and the DSV4
    # window tier: the DEFAULT window-pool size = max(working-set floor, ratio x full-pool tokens).
    # < 1.0 trades retained window-prefix capacity for memory savings; must be in (0, 1]. It is the
    # DSV4 window/full ratio directly. Used only when swa_num_pages_override is None (a runtime
    # rebuild can pin an absolute window instead).
    swa_full_tokens_ratio: float = 0.2
    # Absolute window-pool size in the pool's own pages (usable, dummy excluded); None -> use the
    # ratio default above. A runtime cache rebuild sets this (num_swa_pages) to pin the window
    # regardless of the full anchor; the ratio is the startup default and the fallback.
    swa_num_pages_override: int | None = None
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    # Native checkpoint MTP draft depth. Runtime execution remains disabled until the model
    # and scheduler expose the matching draft/verify/rollback path.
    spec_mtp: int = 0
    max_seq_len_override: int | None = None
    max_extend_tokens: int = 8192
    num_page_override: int | None = None  # if not None, will override the number of pages
    # KV capacity in tokens; resolved into num_page_override by _adjust_config once page_size
    # is final. Mutually exclusive with num_page_override.
    num_token_override: int | None = None
    # Runtime knobs of the multimodal path; the architecture side (vision_config, mrope) lives in ModelConfig.
    mm: MultimodalConfig = field(default_factory=MultimodalConfig)

    def __post_init__(self):
        if self.kv_tiering == "force":
            if self.kv_ram_tokens < 0:
                raise ValueError("--kv-ram-tokens must be >= 0 (0 = the whole context)")
        elif self.kv_tiering not in ("off", "auto"):
            raise ValueError(f"unknown KV RAM tiering mode {self.kv_tiering!r}")
        # "off" and "auto" ignore kv_ram_tokens; auto tiers the whole context.
        if self.moe_backend is None:
            return
        if self.moe_strategy != "auto":
            raise ValueError("moe_backend is the old name of moe_strategy; pass only moe_strategy")
        logger.warning("EngineConfig.moe_backend is deprecated; use moe_strategy")
        object.__setattr__(self, "moe_strategy", self.moe_backend)
        object.__setattr__(self, "moe_backend", None)

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_spec(self) -> ModelSpec:
        return get_model_spec(self.hf_config.architectures[0])

    @cached_property
    def active_encoders(self) -> tuple[EncoderSpec, ...]:
        """The encoder towers this process builds: the family registers them, the checkpoint config carries their section, --mm-disable did not name them."""
        return tuple(
            e
            for e in self.model_spec.encoders
            if getattr(self.hf_config, e.config_key, None) is not None
            and e.kind not in self.mm.disabled_encoders
        )

    @cached_property
    def served_modalities(self) -> frozenset[str]:
        """Modalities this process accepts."""
        return frozenset(m for e in self.active_encoders for m in e.modalities)

    @cached_property
    def model_config(self) -> ModelConfig:
        # the parser sees no section for a tower this process does not build (for the vision tower that also means 1-D rope)
        hf_config = copy.copy(self.hf_config)
        built = {e.config_key for e in self.active_encoders}
        for key in set(ENCODER_SECTIONS) | {e.config_key for e in self.model_spec.encoders}:
            # hasattr: a HF config carries no attribute at all for a section its checkpoint
            # never had, and a GGUF config shim carries no encoder sections at all.
            if key not in built and hasattr(hf_config, key):
                setattr(hf_config, key, None)
        # qwen4exp GGUF only: ``--mmproj`` names a specific external vision tower file,
        # overriding whatever build_gguf_shim auto-discovered next to the model.
        if "vision_config" in built and hasattr(hf_config, "vision_config") and self.mm.mmproj_path:
            from freetoken.models.qwen4_exp.mmproj import read_mmproj_vision_config

            hf_config.vision_config = read_mmproj_vision_config(self.mm.mmproj_path)
        spec = self.model_spec
        quant = checkpoint_quant_config(self.model_path, hf_config, spec)
        set_quant_config(quant)
        model_config = _load_attr(spec.module, spec.parse_config)(hf_config)
        model_config = replace(model_config, quant=quant)
        if model_config.model_type == "qwen3_5_moe" and model_config.native_mtp_layers == 0:
            from freetoken.models.qwen3_5_moe.mtp import has_hf_mtp_weights

            if has_hf_mtp_weights(self.model_path):
                model_config = replace(model_config, native_mtp_layers=1, mtp_expert_resident=True)
        if self.spec_mtp > 0:
            mtp = getattr(getattr(model_config, "qwen4_args", None), "mtp", None)
            if (mtp is None or not mtp.enabled) and model_config.native_mtp_layers == 1:
                from freetoken.models.config import with_mtp_layer

                model_config = with_mtp_layer(model_config, model_config.num_layers)
                if model_config.model_type == "qwen3_5_moe":
                    import os

                    from freetoken.models.qwen3_5_moe.mtp import (
                        MTP_PATH_ENV,
                        is_hf_mtp_head,
                    )

                    if is_hf_mtp_head(os.environ.get(MTP_PATH_ENV)):
                        model_config = replace(model_config, mtp_expert_resident=True)
                if model_config.native_mtp_expert_types is not None:
                    model_config = replace(
                        model_config,
                        mtp_expert_bank=True,
                        gguf_expert_types=model_config.native_mtp_expert_types,
                    )
                return model_config
            if mtp is None or not mtp.enabled:
                raise ValueError(
                    "--spec-mtp > 0 requires a checkpoint that carries native MTP metadata "
                    "(text_config.mtp), which this one does not."
                )
            from freetoken.models.config import with_mtp_layer

            model_config = with_mtp_layer(
                model_config, model_config.num_layers, gguf_expert_types=mtp.gguf_expert_types
            )
        return model_config

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
