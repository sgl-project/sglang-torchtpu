"""GPT-OSS-120B sized for the MMLU-Pro score through lm-eval nightly."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b.accuracy import CONFIG as ACCURACY_CONFIG

CONCURRENCY = 32

CONFIG = deepcopy(ACCURACY_CONFIG)
CONFIG["server_args"].update(
    {
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": 36864,
        "max_total_tokens": 1179648,
        "tpu_compile_bs": [1, 2, 4, 8, 16, CONCURRENCY],
    }
)
