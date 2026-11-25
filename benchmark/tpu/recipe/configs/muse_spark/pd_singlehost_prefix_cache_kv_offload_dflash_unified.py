"""Muse Spark DFlash P/D on the unified Full/SWA KV pool: one TP8 host per engine,
with prefill prefix offload.

pd_singlehost_prefix_cache_kv_offload_dflash.py owns the serving shape; this file
only puts both engines on the unified pool, so the two tests gate the same behavior.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.pd_singlehost_prefix_cache_kv_offload_dflash import (
    DECODE_CONFIG as BASE_DECODE_CONFIG,
)
from benchmark.tpu.recipe.configs.muse_spark.pd_singlehost_prefix_cache_kv_offload_dflash import (  # noqa: F401
    GROUP,
)
from benchmark.tpu.recipe.configs.muse_spark.pd_singlehost_prefix_cache_kv_offload_dflash import (
    PREFILL_CONFIG as BASE_PREFILL_CONFIG,
)

# Overrides the inherited hicache_ratio. A 2K prompt sits inside Muse's window, so
# all of it is SWA; a unified SWA page costs 3 host pages and SWA may use 30% of the
# arena. 2 GB holds the probe and fillers; 1 GB trims the probe's window.
HICACHE_SIZE_GB = 2

PREFILL_CONFIG = deepcopy(BASE_PREFILL_CONFIG)
PREFILL_CONFIG["server_args"].update(
    enable_unified_memory=True, hicache_size=HICACHE_SIZE_GB
)

DECODE_CONFIG = deepcopy(BASE_DECODE_CONFIG)
DECODE_CONFIG["server_args"].update(enable_unified_memory=True)
