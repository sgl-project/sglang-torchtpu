"""Gemma-4-31B-it sized for the thinking-mode AIME26 / GPQA / MMMU-Pro nightly."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gemma4_31b.server import CONFIG as BASE_CONFIG

CONCURRENCY = 64
# 32k thinking budget + prompt and images.
CONTEXT_LENGTH = 32768 + 2048

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/gemma-4-31B-it",
        "served_model_name": "gemma4-31b",
        "reasoning_parser": "gemma4",
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": CONTEXT_LENGTH,
        "tpu_compile_tokens": [512, 2048, 8192],
        "max_total_tokens": CONCURRENCY * CONTEXT_LENGTH,
        # Each request keeps only its 1024-token window plus a page in the SWA
        # pool: 64 x 1152 + one 8192 prefill chunk fits in 4% of the full pool.
        "swa_full_tokens_ratio": 0.04,
        "mem_fraction_static": 0.75,
    }
)
