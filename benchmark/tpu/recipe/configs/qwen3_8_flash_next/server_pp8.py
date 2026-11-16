"""Qwen3.8-Flash-Next-FP8 on one tpu7x host as eight single-device pipeline
stages (PP8), text only. Serve prefill-only traffic (output length 1) to
measure the PP prefill; see server_tp8.py for the context bound.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/qwen3_8_flash_next/server_pp8.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.qwen3_8_flash_next import server_tp8

CONFIG = deepcopy(server_tp8.CONFIG)
CONFIG["server_args"].update(
    {
        "tp_size": 1,
        "ep_size": 1,
        "pp_size": 8,
        # PP rejects the overlap schedule.
        "disable_overlap_schedule": True,
        # max_running_requests / pp_max_micro_batch_size >= pp_size keeps every
        # stage busy.
        "pp_max_micro_batch_size": server_tp8.CONCURRENCY // 8,
        "dist_timeout": 3600,
    }
)
