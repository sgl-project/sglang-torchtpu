"""One-host-per-engine Raiden P/D with the block-major KV layout on both engines.

Both engines allocate every KV pool as one page-major bundle
(SGLANG_TPU_USE_BLOCK_MAJOR_KV), which the Raiden handoff requires on both
sides. The prefill engine adds the Raiden HiCache host tier so the probe can
push a prefix out of HBM and reload it from host RAM in the bundled layout.
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b import pd_singlehost

GROUP = deepcopy(pd_singlehost.GROUP)
PREFILL_CONFIG = deepcopy(pd_singlehost.PREFILL_CONFIG)
DECODE_CONFIG = deepcopy(pd_singlehost.DECODE_CONFIG)

for _config in (PREFILL_CONFIG, DECODE_CONFIG):
    _config["runtime_envs"]["SGLANG_TPU_USE_BLOCK_MAJOR_KV"] = "1"
    if _config["server_args"]["model_path"].startswith("gs://"):
        _config["server_args"]["load_format"] = "runai_streamer"

PREFILL_CONFIG["server_args"].update(
    enable_hierarchical_cache=True,
    # Holds the 2x-pool churn plus the probe target; the host tier evicts LRU too.
    hicache_ratio=int(os.environ.get("HICACHE_RATIO", "3")),
    hicache_io_backend="raiden",
    hicache_mem_layout="page_first_fused",
    hicache_write_policy="write_through",
    # cached_tokens_details in /generate replies needs the cache report.
    enable_cache_report=True,
)
