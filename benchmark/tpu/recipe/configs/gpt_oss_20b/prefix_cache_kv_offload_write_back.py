"""GPT-OSS-20B prefix-cache / Raiden KV-offload gate under write_back.

Same pools as prefix_cache_kv_offload.py; only the write policy differs, so
a prefix reaches host when the device evicts it, not when it is inserted.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_20b/prefix_cache_kv_offload_write_back.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_20b.prefix_cache_kv_offload import (
    CONFIG as WRITE_THROUGH_CONFIG,
)
from benchmark.tpu.recipe.configs.gpt_oss_20b.prefix_cache_kv_offload import (
    PREFILL_TOKENS,
)

CONFIG = deepcopy(WRITE_THROUGH_CONFIG)
CONFIG["server_args"]["hicache_write_policy"] = "write_back"
# One chunk, a quarter of the device pool: evictions take both the copied-ahead
# path and the synchronous one. The default of four chunks would cover the pool.
CONFIG["runtime_env"]["SGLANG_TPU_HICACHE_WRITE_BACK_COPY_AHEAD_TOKENS"] = str(
    PREFILL_TOKENS
)
