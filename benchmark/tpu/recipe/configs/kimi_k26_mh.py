"""TPU serving configuration. Callers copy CONFIG before overriding it."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/Kimi-K2.6",
        "nnodes": 2,
        "tp_size": 16,
        "ep_size": 16,
        "quantization": None,
        "kv_cache_dtype": None,
        "attention_backend": "pallas_tpu_mla",
        "mem_fraction_static": 0.6,
        "max_running_requests": 8,
        "torch_compile_max_bs": 8,
        "context_length": 8192,
        "chunked_prefill_size": 8192,
        "max_prefill_tokens": 8192,
        "tpu_compile_tokens": [64, 128, 256, 512, 1024, 2048, 4096, 8192],
        "trust_remote_code": True,
        "skip_server_warmup": True,
    }
)
