"""GPT-OSS-120B sized for the AIME 2025 score at a 64k high-effort budget nightly."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b.accuracy import CONFIG as ACCURACY_CONFIG

CONCURRENCY = 16

CONFIG = deepcopy(ACCURACY_CONFIG)
CONFIG["server_args"].update(
    {
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": 66560,
        "max_total_tokens": 1064960,
        "tpu_compile_bs": [1, 2, 4, 8, CONCURRENCY],
    }
)
