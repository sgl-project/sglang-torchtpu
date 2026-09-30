"""Qwen3.5-397B-A17B on one 2x2x2 tpu7x slice: tp16/ep16, bf16, vision on.

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
        "model_path": "/weights/Qwen3.5-397B-A17B",
        # node_rank / dist_init_addr come from TPU_WORKER_ID / TPU_WORKER_HOSTNAMES.
        "nnodes": 2,
        "tp_size": 16,
        "ep_size": 16,  # 512 experts at moe_intermediate 1024: whole experts per rank
        "quantization": "unquant",  # bf16 ckpt; blocks quant-method re-detection
        # tpu_platform resolves "auto" to fp8_e4m3 for every RPA backend.
        "kv_cache_dtype": "bfloat16",
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "tpu_compile_tokens": [512, 2048, PREFILL_TOKENS],
        # Uncapped the pool takes all post-weight HBM and decode capture OOMs.
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
