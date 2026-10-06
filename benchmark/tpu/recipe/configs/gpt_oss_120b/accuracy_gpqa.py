"""GPT-OSS-120B sized for the GPQA Diamond score; longer prompts than AIME at the same output budget nightly."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b.accuracy import CONFIG as ACCURACY_CONFIG

CONCURRENCY = 16

CONFIG = deepcopy(ACCURACY_CONFIG)
CONFIG["server_args"].update(
    {
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": 69632,
        "max_total_tokens": 1114112,
        "tpu_compile_bs": [1, 2, 4, 8, CONCURRENCY],
    }
)
