"""Qwen3.8-Flash-Next-FP8 (Qwen4-Exp: hybrid GDN + QSA, hyper-connections, PLE)
on one tpu7x host, tp8, text only.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/qwen3_8_flash_next/server_tp8.py

TPU has no QSA kernel yet, so the full-attention layers run dense
(sglang_tpu.hooks.qwen4_exp); that equals QSA while every sequence fits the
indexer budget (2048 tokens), which bounds the context here.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONCURRENCY = 32
PREFILL_TOKENS = 2048
# QSA selects indexer_budget tokens per query row; dense attention matches it
# up to that length.
CONTEXT_LENGTH = 2048

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/Qwen3.8-Flash-Next-FP8",
        "served_model_name": "Qwen/Qwen3.8-Flash-Next-FP8",
        "tp_size": 8,
        "quantization": "fp8",
        # 2 KV heads over 8 ranks: keep KV unpacked.
        "kv_cache_dtype": "bfloat16",
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "tpu_compile_tokens": [512, PREFILL_TOKENS],
        "max_total_tokens": CONCURRENCY * CONTEXT_LENGTH,
        "max_mamba_cache_size": CONCURRENCY,
        "mem_fraction_static": 0.75,
        "watchdog_timeout": 3600,
        "language_model_only": True,
        "trust_remote_code": True,
        "tpu_compile_tracer": "direct",
    }
)
CONFIG["runtime_env"].update(
    {
        # Logs every XLA program compiled or loaded after warmup.
        "SGLANG_TPU_CHECK_RECOMPILATION": "1",
        "TORCH_LOGS": "-dynamo",
    }
)
