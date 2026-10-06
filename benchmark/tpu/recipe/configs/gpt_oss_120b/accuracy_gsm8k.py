"""GPT-OSS-120B sized for the full 1,319-question GSM8K score at low effort nightly."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b.accuracy import CONFIG as ACCURACY_CONFIG

CONCURRENCY = 32

CONFIG = deepcopy(ACCURACY_CONFIG)
CONFIG["server_args"].update(
    {
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": 8192,
        "max_total_tokens": 262144,
        "tpu_compile_bs": [8, 16, CONCURRENCY],
    }
)
