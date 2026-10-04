"""Qwen3.5-397B-A17B-FP8 on one tpu7x host: tp8/ep8, fp8 KV, vision on.

    python3 -m benchmark.tpu.launch_server \
        --config recipe/configs/qwen3_5_397b/server.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

# Also the recurrent-state slot count, which bounds in-flight requests here.
CONCURRENCY = 128
CONTEXT_LENGTH = 32768 + 8192  # 32k thinking budget + prompt and images
PREFILL_TOKENS = 8192

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/Qwen3.5-397B-A17B-FP8",
        "served_model_name": "Qwen/Qwen3.5-397B-A17B-FP8",
        # 397B of FP8 weights is 47 GB per core on 8 cores; bf16 needs two hosts.
        "tp_size": 8,
        "ep_size": 8,
        "quantization": "fp8",
        "kv_cache_dtype": "fp8_e4m3",
        "tpu_fp8_runtime": "channelwise",
        "page_size": 256,
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "tpu_compile_tokens": [512, 2048, PREFILL_TOKENS],
        # Uncapped the pool takes all post-weight HBM and decode capture OOMs.
        # 1.31M fp8 tokens on 8 cores is the same 19 GB per core as bf16 on 16.
        "max_total_tokens": 1310720,
        "mem_fraction_static": 0.85,
        "trust_remote_code": True,
        "watchdog_timeout": 3600,  # graph captures outlast the 300 s default
        "reasoning_parser": "qwen3",
        # 1280*28*28 px = 3920 ViT patches; uncapped one page exceeds the 8k chunk.
        "mm_process_config": {"image": {"min_pixels": 65536, "max_pixels": 1003520}},
    }
)
CONFIG["runtime_env"].update(
    {
        # Square patch-count buckets warmed at startup; one ViT call carries all
        # of a request's images, so the ladder must reach the multi-image total.
        "SGLANG_TPU_VIT_PRECOMPILE_SIZES": "1024,2304,4096,6400,9216,12544,16384",
    }
)
