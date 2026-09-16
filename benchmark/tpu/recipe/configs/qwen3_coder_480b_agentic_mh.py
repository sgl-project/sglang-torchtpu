"""TPU recipe settings; callers copy dictionaries before overriding them."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.qwen3_coder_480b.nightly.serve import (
    CONFIG as BASE_CONFIG,
)

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/Qwen3-Coder-480B-A35B-Instruct-FP8",
        "max_running_requests": 32,
        "torch_compile_max_bs": 16,
        "tpu_compile_tokens": [4096],
        "tpu_compile_bs": [1, 8, 16],
    }
)
