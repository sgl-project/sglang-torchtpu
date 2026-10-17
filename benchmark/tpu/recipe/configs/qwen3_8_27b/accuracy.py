"""Qwen3.8-27B (bf16 dense hybrid GDN) on one tpu7x host, tp8, vision on,
sized for the thinking-mode GPQA / AIME / MMMU-Pro nightly.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/qwen3_8_27b/accuracy.py

48 of 64 layers are GDN linear attention with a fixed-size recurrent state,
16 are full attention with 4 KV heads.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONCURRENCY = 64
# 80k thinking budget + prompt and images.
CONTEXT_LENGTH = 81920 + 8192
PREFILL_TOKENS = 8192

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/Qwen3.8-27B",
        "served_model_name": "Qwen/Qwen3.8-27B",
        "tp_size": 8,
        # "unquant", not None: it also sets _quantization_explicitly_unset, so
        # the model overrides cannot re-detect a quant method.
        "quantization": "unquant",
        "kv_cache_dtype": "bfloat16",
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "tpu_compile_tokens": [512, 2048, PREFILL_TOKENS],
        # Paged, so this bounds the sum of live contexts, not each one. At 16k
        # per slot GPQA filled the pool with 37 of 64 running and retracted
        # constantly; 32k per slot is 32 GB of the 88 GB free per core.
        "max_total_tokens": CONCURRENCY * 32768,
        "mem_fraction_static": 0.75,
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
        # The base +recompiles makes dynamo reprint its guard list on every
        # recompile. Serving-time recompiles still fail the scheduler.
        "TORCH_LOGS": "-dynamo",
    }
)
