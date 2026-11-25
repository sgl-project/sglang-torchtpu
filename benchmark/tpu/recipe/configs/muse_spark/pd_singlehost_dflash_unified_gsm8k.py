"""Muse Spark DFlash P/D on the unified Full/SWA KV pool for a full GSM8K score:
one TP8 host per engine.

pd_singlehost_prefix_cache_kv_offload_dflash.py owns the engine shape; this file
drops its bounded prefill pool and prefill host offload, and puts both engines on
the unified pool, so the score reflects P/D transfer and the unified pool alone.
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.pd_singlehost_prefix_cache_kv_offload_dflash import (  # noqa: F401
    GROUP,
)
from benchmark.tpu.recipe.configs.muse_spark.pd_singlehost_prefix_cache_kv_offload_dflash import (
    _COMMON_CONFIG,
)
from benchmark.tpu.recipe.configs.muse_spark.pd_singlehost_prefix_cache_kv_offload_dflash import (
    DECODE_CONFIG as BASE_DECODE_CONFIG,
)

PREFILL_CONFIG = deepcopy(_COMMON_CONFIG)
PREFILL_CONFIG["server_args"].update(
    port=int(os.environ.get("PREFILL_PORT", "30000")),
    disaggregation_mode="prefill",
    disaggregation_bootstrap_port=int(os.environ.get("BOOTSTRAP_PORT", "8998")),
    enable_unified_memory=True,
)

DECODE_CONFIG = deepcopy(BASE_DECODE_CONFIG)
DECODE_CONFIG["server_args"].update(enable_unified_memory=True)
