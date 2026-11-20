"""DeepSeek-V4.1-Flash + DSpark served for AgentX on one tpu7x-8 host (DP8/EP8).

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_dspark_c32.py

The AgentX sizing the canonical harness has applied on top of
deepseek_v41_flash_dspark.py (combined-1 canonical: c8 7,314 tok/s/chip @ p90
interactivity 40.9, c32 24,140 @ 30.0). It is the base of the per-concurrency
recipes deepseek_v41_flash_dspark_c{4,8,16,32,64,128,256}.py, which override
only what their concurrency needs.

Global values are divided by attn_dp_size (8) per rank.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.deepseek_v41_flash_dspark import (
    CONFIG as DSPARK_CONFIG,
)

CONFIG = deepcopy(DSPARK_CONFIG)
CONFIG["server_args"].update(
    {
        # AgentX prompts reach ~1M tokens (avg ~140k).
        "context_length": 1048576,
        # Radix cache + encoder SWA bounded replay: most of each prompt repeats
        # the previous turn. Needs mixed chunk off and >= 384 prefill tokens
        # per rank.
        "disable_radix_cache": False,
        "enable_encoder_swa_bounded_replay": True,
        # 1024 prefill tokens per rank and step.
        "chunked_prefill_size": 8192,
        "max_prefill_tokens": 8192,
        # 16 per rank. The per-concurrency recipes raise it at c >= 64.
        "max_running_requests": 128,
        "schedule_policy": "fcfs",
    }
)


def with_max_running_requests(n: int) -> dict:
    """This config with a different global max_running_requests."""
    config = deepcopy(CONFIG)
    config["server_args"]["max_running_requests"] = n
    return config
