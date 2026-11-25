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

# A unified SWA page costs num_swa_groups (3) host pages and SWA may use 30% of the
# arena, so ratio 16 matches the separate twin's SWA host capacity (193 pages at 3);
# at 3 the probe's SWA window is trimmed and the host reload comes back empty.
HICACHE_RATIO = 16

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(enable_unified_memory=True, hicache_ratio=HICACHE_RATIO)
