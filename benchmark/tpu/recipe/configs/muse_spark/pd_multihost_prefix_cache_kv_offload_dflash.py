"""Muse Spark DFlash P/D: two hosts per engine, with prefill prefix offload.

Every pod launches its slice-local role from CI_PD_ROLE. The bounded prefill
pool makes the cache gate force an actual host reload.
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.server import CONFIG as SERVER_CONFIG

ROLE = os.environ.get("CI_PD_ROLE", "prefill")

CONFIG = deepcopy(SERVER_CONFIG)
CONFIG["server_args"].update(
    model_path="/weights/Muse-Spark-Open-816B-A42B-NVFP4",
    served_model_name="meta-models/Muse-Spark-Open-816B-A42B-NVFP4",
    host="0.0.0.0",
    nnodes=2,
    tp_size=16,
    context_length=8192,
    mem_fraction_static=0.85,
    speculative_algorithm="DFLASH",
    speculative_draft_model_path="/weights/Muse-Spark-Open-DFlash2",
    speculative_num_draft_tokens=8,
    disaggregation_mode=ROLE,
    disaggregation_transfer_backend="raiden",
    enable_cache_report=True,
    skip_server_warmup=True,
    sampling_backend="pytorch",
)
if ROLE == "prefill":
    CONFIG["server_args"].update(
        max_total_tokens=8192,
        # The server's 0.125 leaves 1,024 SWA tokens, below the 2,047-token
        # window plus a page that startup requires of the SWA pool.
        swa_full_tokens_ratio=0.5,
        enable_hierarchical_cache=True,
        hicache_io_backend="raiden",
        hicache_mem_layout="page_first_fused",
        hicache_write_policy="write_through",
        hicache_ratio=3,
    )
CONFIG["runtime_env"].update(
    FORCE_SINGLE_HOST=None,
    SGLANG_RAIDEN_RESHARD="0",
    SGLANG_SPEC_TP_SYNC="rng",
    SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER="1",
)
