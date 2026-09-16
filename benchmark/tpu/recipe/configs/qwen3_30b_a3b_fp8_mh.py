"""TPU serving configuration. Callers copy CONFIG before overriding it."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/Qwen3-30B-A3B-FP8",
        "ep_size": 8,
        "quantization": None,
        "kv_cache_dtype": None,
        "nnodes": 2,
        "tp_size": 16,
    }
)
