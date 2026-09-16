"""TPU serving configuration. Callers copy CONFIG before overriding it."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/DeepSeek-R1-0528",
        "nnodes": 2,
        "tp_size": 16,
        "ep_size": 16,
        "attention_backend": "pallas_tpu_mla",
        "kv_cache_dtype": None,
        "mem_fraction_static": 0.85,
        "max_running_requests": 128,
        "torch_compile_max_bs": 128,
        "page_size": 64,
        "context_length": 40960,
        "chunked_prefill_size": 8192,
        "max_prefill_tokens": 8192,
        "tpu_compile_tokens": [8192],
        "tpu_compile_bs": [64, 128],
        "reasoning_parser": "deepseek-r1",
        "weight_loader_drop_cache_after_load": True,
    }
)
