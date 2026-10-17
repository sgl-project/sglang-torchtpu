"""Qwen3.8-27B BF16 on one TPU host, tp8, vision on, sized for the full-set
thinking-mode MMMU-Pro nightly. The text-only GPQA / AIME recipe is accuracy.py.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/qwen3_8_27b/mmmu_pro.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

# Also the recurrent-state slot count, which bounds in-flight requests here.
CONCURRENCY = 32
CONTEXT_LENGTH = 32768 + 8192  # 32k thinking budget + prompt and images
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
        "page_size": 64,
        "max_running_requests": CONCURRENCY,
        "max_mamba_cache_size": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "tpu_compile_tokens": [512, 2048, PREFILL_TOKENS],
        # 16k tokens per request, not CONTEXT_LENGTH: the full 1,730-question run
        # peaked at 78% of this pool with no retractions (8 GB of KV per core).
        "max_total_tokens": 524288,
        "mem_fraction_static": 0.75,
        "watchdog_timeout": 3600,  # graph captures outlast the 300 s default
        "reasoning_parser": "qwen3",
        # Match the image budget used by the Qwen3.5 MMMU-Pro accuracy recipe.
        "mm_process_config": {"image": {"min_pixels": 65536, "max_pixels": 1003520}},
    }
)
CONFIG["runtime_env"].update(
    {
        # A request may carry several images through one vision encoder call.
        "SGLANG_TPU_VIT_PRECOMPILE_SIZES": "1024,2304,4096,6400,9216,12544,16384",
        # The base +recompiles makes dynamo reprint its guard list on every
        # recompile. Serving-time recompiles still fail the scheduler.
        "TORCH_LOGS": "-dynamo",
    }
)
