"""GPT-OSS-20B sized for the high-reasoning GPQA / AIME25 / MMLU nightly at
concurrency 128.

server.py in this directory owns how the model is served; this file changes
only what the eval needs.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_20b/accuracy.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_20b.server import CONFIG as SERVER_CONFIG

CONCURRENCY = 128
CONTEXT_LENGTH = 131072
PREFILL_TOKENS = 4096

CONFIG = deepcopy(SERVER_CONFIG)
CONFIG["server_args"].update(
    {
        # Where the scores were taken, and so where they stay: page 128,
        # mem fraction 0.8, the dynamo tracer.
        "page_size": 128,
        "mem_fraction_static": 0.8,
        "tpu_compile_tracer": "dynamo",
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "tpu_compile_bs": [1, 8, 32, 64, CONCURRENCY],
        "tpu_compile_tokens": [PREFILL_TOKENS],
        "context_length": CONTEXT_LENGTH,
        "max_total_tokens": 64 * CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "disable_overlap_schedule": True,
        # The eval's prompts share no prefix, so the radix tree only costs
        # memory, and the scores were taken without it.
        "disable_radix_cache": True,
        "reasoning_parser": "gpt-oss",
        # The evaluator asks for the full 131072-token budget before counting
        # prompt tokens; cap generation to the remaining context instead.
        "allow_auto_truncate": True,
    }
)
