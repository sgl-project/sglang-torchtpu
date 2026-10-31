"""Muse Spark 1P1D under overload: one TP8 host per engine, 256K, multimodal.

The prefill keeps a prefix cache with host KV offload. Both engines export
metrics so the stress test can prove overload and drain. The mini-LB runs the
session-affinity policy, whose guardrails shed new requests with 429 once an
engine's queue passes its limit; the TPU P/D engines reject
--max-queued-requests, so the router is the only place that can shed.
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.server import CONFIG as SERVER_CONFIG

GROUP = {
    "hosts": 2,
    "prefill_engines": 1,
    "decode_engines": 1,
    "router_args": ["--mini-lb-policy", "session_affinity"],
}

_COMMON_CONFIG = deepcopy(SERVER_CONFIG)
_COMMON_CONFIG["server_args"].update(
    model_path="/weights/Muse-Spark-Open-816B-A42B-NVFP4",
    served_model_name="meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    host="0.0.0.0",
    kv_cache_dtype="fp8_e4m3",
    enable_multimodal=True,
    disaggregation_transfer_backend="raiden",
    enable_metrics=True,
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
    # Image and video serving loads the TPU Muse processor (sglang_tpu/README.md).
    SGLANG_EXTERNAL_MM_MODEL_ARCH="MuseSparkForConditionalGeneration",
    SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE="sglang_tpu.multimodal",
)

PREFILL_CONFIG = deepcopy(_COMMON_CONFIG)
PREFILL_CONFIG["server_args"].update(
    port=int(os.environ.get("PREFILL_PORT", "30000")),
    disaggregation_mode="prefill",
    disaggregation_bootstrap_port=int(os.environ.get("BOOTSTRAP_PORT", "8998")),
    enable_hierarchical_cache=True,
    hicache_io_backend="raiden",
    hicache_mem_layout="page_first_fused",
    hicache_write_policy="write_through",
    hicache_ratio=2,
)

DECODE_CONFIG = deepcopy(_COMMON_CONFIG)
DECODE_CONFIG["server_args"].update(
    port=int(os.environ.get("DECODE_PORT", "30000")),
    disaggregation_mode="decode",
)
