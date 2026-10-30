"""Muse Spark DFlash P/D: one TP8 host per engine, with prefill prefix offload.

The bounded prefill pool makes the cache gate force an actual host reload.
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.server import CONFIG as SERVER_CONFIG

GROUP = {"hosts": 2, "prefill_engines": 1, "decode_engines": 1}

_COMMON_CONFIG = deepcopy(SERVER_CONFIG)
_COMMON_CONFIG["server_args"].update(
    model_path="/weights/Muse-Spark-Open-816B-A42B-NVFP4",
    served_model_name="meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    host="0.0.0.0",
    context_length=8192,
    mem_fraction_static=0.85,
    kv_cache_dtype="fp8_e4m3",
    speculative_algorithm="DFLASH",
    speculative_draft_model_path="/weights/Muse-Spark-Open-DFlash2",
    speculative_num_draft_tokens=8,
    disaggregation_transfer_backend="raiden",
    enable_cache_report=True,
    skip_server_warmup=True,
    sampling_backend="pytorch",
)
_COMMON_CONFIG["runtime_env"].update(
    # CI keeps the compile caches in the pod's 64Gi /dev/shm.
    ENABLE_COMPILE_CACHE="0",
    LB_PORT=os.environ.get("LB_PORT", "8000"),
    SGLANG_RAIDEN_KV_PORT=os.environ.get("SGLANG_RAIDEN_KV_PORT", "31000"),
    SGLANG_RAIDEN_RESHARD="0",
    SGLANG_SPEC_TP_SYNC="rng",
    SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER="1",
)

PREFILL_CONFIG = deepcopy(_COMMON_CONFIG)
PREFILL_CONFIG["server_args"].update(
    port=int(os.environ.get("PREFILL_PORT", "30000")),
    disaggregation_mode="prefill",
    disaggregation_bootstrap_port=int(os.environ.get("BOOTSTRAP_PORT", "8998")),
    max_total_tokens=8192,
    # Muse's 2,048-token window outlasts a 2K probe prompt, so every prompt
    # token stays live. A smaller SWA pool stalled the probe's eviction chunks.
    swa_full_tokens_ratio=1.0,
    enable_hierarchical_cache=True,
    hicache_io_backend="raiden",
    hicache_mem_layout="page_first_fused",
    hicache_write_policy="write_through",
    hicache_ratio=3,
)

DECODE_CONFIG = deepcopy(_COMMON_CONFIG)
DECODE_CONFIG["server_args"].update(
    port=int(os.environ.get("DECODE_PORT", "30000")),
    disaggregation_mode="decode",
)
