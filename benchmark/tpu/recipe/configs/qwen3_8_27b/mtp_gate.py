"""Qwen3.8-27B MTP (3/1/4) on the experimental TPU spec worker, one tpu7x host, tp8,
sized for the PR gate: small batch and token buckets so the precompile stays short.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/qwen3_8_27b/mtp_gate.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONCURRENCY = 16
PREFILL_TOKENS = 2048

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/Qwen3.8-27B",
        "served_model_name": "Qwen/Qwen3.8-27B",
        "tp_size": 8,
        "quantization": "unquant",
        "kv_cache_dtype": "bfloat16",
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": 4096,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "tpu_compile_tokens": [512, PREFILL_TOKENS],
        "max_total_tokens": CONCURRENCY * 4096,
        "max_mamba_cache_size": CONCURRENCY,
        "mem_fraction_static": 0.75,
        "watchdog_timeout": 3600,
        "language_only": True,
        "trust_remote_code": True,
        "tpu_compile_tracer": "direct",
        "speculative_algorithm": "NEXTN",
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
    }
)
CONFIG["runtime_env"].update(
    {
        "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER": "1",
        "SGLANG_TPU_HOST_PADDING": "1",
        "SGLANG_SPEC_TP_SYNC": "rng",
        "SGLANG_RAGGED_VERIFY_MODE": "static",
        # Logs every XLA program compiled or loaded after warmup.
        "SGLANG_TPU_CHECK_RECOMPILATION": "1",
        "TORCH_LOGS": "-dynamo",
    }
)
