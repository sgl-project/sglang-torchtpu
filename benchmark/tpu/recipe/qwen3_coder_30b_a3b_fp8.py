"""TPU serving configuration. Callers copy CONFIG before overriding it."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        "tp_size": 8,
        "ep_size": 8,
        "context_length": 8192,
        "max_total_tokens": 262144,
        "mem_fraction_static": 0.3,
    }
)
