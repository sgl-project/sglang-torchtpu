"""Single-chip Qwen3 config with small shapes for the compile-tracer e2e test."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _BASE_CONFIG

INPUT_TOKENS = 64
OUTPUT_TOKENS = 64
CONCURRENCY = 1

CONFIG = deepcopy(_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "Qwen/Qwen3-0.6B",
        "tp_size": 1,
        "page_size": 64,
        "chunked_prefill_size": INPUT_TOKENS,
        "context_length": INPUT_TOKENS + OUTPUT_TOKENS,
        "max_prefill_tokens": INPUT_TOKENS,
        "max_total_tokens": INPUT_TOKENS + OUTPUT_TOKENS,
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "enable_torch_compile": True,
        "tpu_compile_tokens": [16],
        "tpu_compile_bs": [1],
    }
)
CONFIG["runtime_env"].update(
    {
        "TPU_VISIBLE_CHIPS": "0",
        "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
        "TPU_HOST_BOUNDS": "1,1,1",
    }
)
