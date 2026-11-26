"""Muse Spark on one host with DFlash, the prefix cache and FP8 KV offload, on the
unified Full/SWA KV pool.

singlehost_prefix_cache_kv_offload_dflash.py owns the serving shape; this file only
adds the unified pool, so the two tests gate the same behavior.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/muse_spark/singlehost_prefix_cache_kv_offload_dflash_unified.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.singlehost_prefix_cache_kv_offload_dflash import (
    CONFIG as BASE_CONFIG,
)

# Overrides the inherited hicache_ratio. A 2K prompt sits inside Muse's window, so
# all of it is SWA; a unified SWA page costs 3 host pages and SWA may use 30% of the
# arena. 2 GB holds the probe and fillers; 1 GB trims the probe's window.
HICACHE_SIZE_GB = 2

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(enable_unified_memory=True, hicache_size=HICACHE_SIZE_GB)
