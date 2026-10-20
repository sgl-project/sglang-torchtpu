"""DeepSeek-V4-Flash (native packed-FP4 W4A8 MoE + DSV4 hybrid attention) on one tpu7x-8 host, tp8.

Sample usage (from the repo root):
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v4_flash.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/deepseek-v4-flash",
        "served_model_name": "deepseek-v4-flash",
        "trust_remote_code": True,
        "tp_size": 8,
        "attention_backend": "dsv4",
        "page_size": 64,
        "kv_cache_dtype": "bfloat16",
        "context_length": 32768,
        "max_total_tokens": 131072,
        "mem_fraction_static": 0.70,
        "chunked_prefill_size": 4096,
        "max_prefill_tokens": 4096,
        "max_running_requests": 64,
        "torch_compile_max_bs": 64,
        "tpu_compile_tokens": [512, 1024, 2048, 4096],
        "tpu_compile_bs": [1, 2, 4, 8, 16, 32, 64],
    }
)
CONFIG["runtime_env"].update(
    {
        "SGLANG_DSV4_FP4_EXPERTS": "1",
        "SGLANG_DSV4_FP4_DEQUANT": "0",
    }
)
