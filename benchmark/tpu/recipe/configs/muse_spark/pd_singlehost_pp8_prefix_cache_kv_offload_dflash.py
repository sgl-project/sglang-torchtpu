"""Muse Spark DFlash P/D: single-host PP8 prefill + TP8 decode with KV offload.

Prefill runs PP=8, TP=1, EP=1 (asymmetric 7,8,8,8,8,8,8,7 partition, stock
Pallas gmm_v2 sub-512 NVFP4, host-quantized BF16->FP8 boundary experts, FP32
PPWave + BF16 PPAuxWave DFlash2 capture relay, and Raiden L2 HiCache). Decode
runs TP=8, EP=1, PP=1 and pulls named layer pools over Raiden pipeline transfer.
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.server import CONFIG as TP8_CONFIG
from benchmark.tpu.recipe.configs.muse_spark.server_pp8 import CONFIG as PP8_CONFIG

GROUP = {"hosts": 2, "prefill_engines": 1, "decode_engines": 1}

MODEL_PATH = os.environ.get(
    "MUSE_SPARK_MODEL", "/weights/Muse-Spark-Open-816B-A42B-NVFP4"
)
DRAFT_MODEL = os.environ.get(
    "MUSE_SPARK_DRAFT_MODEL", "RadixArk/Muse-Spark-Open-DFlash2"
)

_COMMON_ARGS = {
    "model_path": MODEL_PATH,
    "served_model_name": "meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    "host": "0.0.0.0",
    "context_length": 8192,
    "mem_fraction_static": 0.85,
    # Raiden's PP8 -> TP8 pipeline transfer gives each decode rank 1 KV head,
    # and token-major FP8 packing needs an even KV head count per rank.
    "kv_cache_dtype": "bfloat16",
    "speculative_algorithm": "DFLASH",
    "speculative_draft_model_path": DRAFT_MODEL,
    "speculative_num_draft_tokens": 8,
    "disaggregation_transfer_backend": "raiden",
    "enable_cache_report": True,
    "enable_metrics": True,
    "skip_server_warmup": True,
    "sampling_backend": "pytorch",
    "dist_timeout": 3600,
}

_COMMON_ENVS = {
    "FORCE_SINGLE_HOST": None,
    # CI keeps compile caches in the pod's 64Gi /dev/shm.
    "ENABLE_COMPILE_CACHE": "0",
    "LB_PORT": os.environ.get("LB_PORT", "8000"),
    "HEALTH_TIMEOUT": os.environ.get("WAIT_SECS", "5400"),
    "SGLANG_RAIDEN_KV_PORT": os.environ.get("SGLANG_RAIDEN_KV_PORT", "31000"),
    "SGLANG_SPEC_TP_SYNC": "rng",
    "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER": "1",
}

PREFILL_CONFIG = deepcopy(PP8_CONFIG)
PREFILL_CONFIG["server_args"].update(
    **_COMMON_ARGS,
    port=int(os.environ.get("PREFILL_PORT", "30000")),
    disaggregation_mode="prefill",
    disaggregation_bootstrap_port=int(os.environ.get("BOOTSTRAP_PORT", "8998")),
    max_total_tokens=8192,
    # Muse's 2,048-token window outlasts a 2K probe prompt, so every prompt
    # token stays live without stalling the KV offload probe's eviction chunks.
    swa_full_tokens_ratio=1.0,
    enable_hierarchical_cache=True,
    hicache_io_backend="raiden",
    hicache_mem_layout="page_first_fused",
    hicache_write_policy="write_through",
    hicache_ratio=3,
)
PREFILL_CONFIG["runtime_env"].update(_COMMON_ENVS)

DECODE_CONFIG = deepcopy(TP8_CONFIG)
DECODE_CONFIG["server_args"].update(
    **_COMMON_ARGS,
    port=int(os.environ.get("DECODE_PORT", "30000")),
    disaggregation_mode="decode",
    disaggregation_decode_retraction_backup="cpu_tensor",
)
DECODE_CONFIG["runtime_env"].update(
    **_COMMON_ENVS,
    # A non-PP decode must name its layer pools to pull from a PP producer.
    SGLANG_RAIDEN_PIPELINE_TRANSFER="1",
)
