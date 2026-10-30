"""Muse Spark on one host with DFlash, the prefix cache and FP8 KV offload.

The bounded device pool makes the cache gate force an actual host reload.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.server import CONFIG as SERVER_CONFIG

CONFIG = deepcopy(SERVER_CONFIG)
CONFIG["server_args"].update(
    model_path="/weights/Muse-Spark-Open-816B-A42B-NVFP4",
    served_model_name="meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    context_length=8192,
    mem_fraction_static=0.85,
    kv_cache_dtype="fp8_e4m3",
    max_total_tokens=8192,
    # Muse's 2,048-token window outlasts a 2K probe prompt, so every prompt
    # token stays live. A smaller SWA pool stalled the probe's eviction chunks.
    swa_full_tokens_ratio=1.0,
    speculative_algorithm="DFLASH",
    speculative_draft_model_path="/weights/Muse-Spark-Open-DFlash2",
    speculative_num_draft_tokens=8,
    enable_hierarchical_cache=True,
    hicache_io_backend="raiden",
    hicache_mem_layout="page_first_fused",
    hicache_write_policy="write_through",
    hicache_ratio=3,
    enable_cache_report=True,
    skip_server_warmup=True,
    sampling_backend="pytorch",
)
CONFIG["runtime_env"].update(
    SGLANG_SPEC_TP_SYNC="rng",
    SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER="1",
)
