#!/usr/bin/env python3
"""Standalone test script to trace engine initialization."""

import os
import sys
import traceback
import time

os.environ["FREETOKEN_LOG_LEVEL"] = "DEBUG"


def trace(msg):
    print(f"[{time.time():.3f}] {msg}", flush=True)


trace("Starting imports")
from freetoken.engine.config import EngineConfig

trace("EngineConfig imported")
from freetoken.engine.engine import Engine

trace("Engine imported")
from freetoken.distributed import DistributedInfo

trace("DistributedInfo imported")
from freetoken.mm.config import MultimodalConfig, ENCODER_KINDS

trace("MultimodalConfig imported")
import torch

trace("torch imported")

trace("Creating config")
config = EngineConfig(
    model_path="/models/Qwen3.8-Flash-Next-NVFP4-Radix",
    tp_info=DistributedInfo(rank=0, size=1),
    dtype=torch.bfloat16,
    kv_format="turbo4",
    moe_strategy="offload",
    moe_cache_auto=True,
    max_seq_len_override=16384,
    cuda_graph_max_bs=0,
    mm=MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)),
    moe_prefill_overlap=False,
    use_dummy_weight=True,
)
trace("Config created")

trace("Accessing config.model_config...")
mc = config.model_config
trace(f"model_config: {mc}")

trace("Creating Engine...")
engine = Engine(config)
trace("Engine created successfully!")
