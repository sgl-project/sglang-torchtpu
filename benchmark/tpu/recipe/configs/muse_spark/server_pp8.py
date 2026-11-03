"""Muse Spark (816B-A42B NVFP4) on one tpu7x host, 4 chips x 2 cores = pp8.

Pipeline-parallel single-host serving (pp 8, tp 1, ep 1): each rank holds all
256 experts of 7-8 decoder layers and forwards an fp32 residual stream via
PPWave. NVFP4 experts (group_size=16) run through stock Pallas gmm_v2 (W4A16);
the BF16 first/last layers quantize online to FP8 on host during load.
Partitioning layers as 7,8,8,8,8,8,8,7 keeps the two BF16->FP8 layers (0 and
61) plus embed_tokens and lm_head balanced at ~64 GiB per stage.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/muse_spark/server_pp8.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.server import CONFIG as SERVER_CONFIG

CONFIG = deepcopy(SERVER_CONFIG)
CONFIG["server_args"].update(
    {
        "tp_size": 1,
        "pp_size": 8,
        # Fused-RS requires TP/EP > 1; auto selects stock Pallas gmm_v2 at PP8/EP1.
        "moe_runner_backend": "auto",
        "mem_fraction_static": 0.90,
        "disable_overlap_schedule": True,
        "dist_timeout": 3600,
    }
)
CONFIG["runtime_env"].update(
    {
        "SGLANG_PP_LAYER_PARTITION": "7,8,8,8,8,8,8,7",
        "SGLANG_WARMUP_TIMEOUT": "3600",
    }
)
