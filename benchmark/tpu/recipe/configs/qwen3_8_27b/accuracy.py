"""Qwen3.8-27B (bf16 dense hybrid GDN) on one tpu7x host, tp8, text only,
sized for the thinking-mode GPQA / AIME nightly.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/qwen3_8_27b/accuracy.py

48 of 64 layers are GDN linear attention with a fixed-size recurrent state,
16 are full attention with 4 KV heads.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONCURRENCY = 64
# 80k thinking budget + prompt.
CONTEXT_LENGTH = 81920 + 4096
PREFILL_TOKENS = 4096

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
        "tpu_compile_tokens": [512, PREFILL_TOKENS],
        # Paged, so this bounds the sum of live contexts, not each one. At 16k
        # per slot GPQA filled the pool with 37 of 64 running and retracted
        # constantly; 32k per slot is 32 GB of the 88 GB free per core.
        "max_total_tokens": CONCURRENCY * 32768,
        "mem_fraction_static": 0.75,
        "watchdog_timeout": 3600,  # graph captures outlast the 300 s default
        "reasoning_parser": "qwen3",
        # Text only. The ViT never executes and must not precompile.
        "language_only": True,
    }
)
CONFIG["runtime_env"].update({"SGLANG_TPU_VIT_PRECOMPILE_SIZES": ""})
