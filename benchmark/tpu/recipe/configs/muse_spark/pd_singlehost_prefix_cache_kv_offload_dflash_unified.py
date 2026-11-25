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

# A unified SWA page costs num_swa_groups (3) host pages and SWA may use 30% of the
# arena, so ratio 16 matches the separate twin's SWA host capacity (193 pages at 3);
# at 3 the probe's SWA window is trimmed and the host reload comes back empty.
HICACHE_RATIO = 16

PREFILL_CONFIG = deepcopy(BASE_PREFILL_CONFIG)
PREFILL_CONFIG["server_args"].update(
    enable_unified_memory=True, hicache_ratio=HICACHE_RATIO
)

DECODE_CONFIG = deepcopy(BASE_DECODE_CONFIG)
DECODE_CONFIG["server_args"].update(enable_unified_memory=True)
