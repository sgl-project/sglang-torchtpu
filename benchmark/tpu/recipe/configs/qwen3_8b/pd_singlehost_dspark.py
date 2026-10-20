"""Qwen3-8B DSpark on the experimental TPU spec worker, Raiden P/D with one
tpu7x host per engine, sized for a CI gate.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/qwen3_8b/pd_singlehost_dspark.py:PREFILL_CONFIG
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.base import (
    _FP8_BASE_CONFIG,
    PD_SINGLEHOST_GROUP,
    pd_singlehost_role_configs,
)

CONCURRENCY = 16
PREFILL_TOKENS = 2048
DRAFT_MODEL = os.environ.get("DRAFT_MODEL") or "/weights/dspark_qwen3_8b_block7"

_CONFIG = deepcopy(_FP8_BASE_CONFIG)
_CONFIG["server_args"].update(
    {
        "model_path": "/weights/Qwen3-8B",
        "served_model_name": "Qwen/Qwen3-8B",
        "tp_size": 8,
        "quantization": None,
        "kv_cache_dtype": "auto",
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "context_length": 4096,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "tpu_compile_tokens": [512, PREFILL_TOKENS],
        "max_total_tokens": CONCURRENCY * 4096,
        "mem_fraction_static": 0.3,
        "watchdog_timeout": 3600,
        "trust_remote_code": True,
        "tpu_compile_tracer": "direct",
        "speculative_algorithm": "DSPARK",
        "speculative_draft_model_path": DRAFT_MODEL,
    }
)
_CONFIG["runtime_env"].update(
    {
        "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER": "1",
        "SGLANG_TPU_HOST_PADDING": "1",
        "SGLANG_SPEC_TP_SYNC": "rng",
        "SGLANG_RAGGED_VERIFY_MODE": "static",
        # Logs every XLA program compiled after warmup, with its source line.
        "SGLANG_TPU_CHECK_RECOMPILATION": "1",
        "SGLANG_TPU_CHECK_RECOMPILATION_SITES": "1",
        "TORCH_LOGS": "-dynamo",
    }
)

GROUP = deepcopy(PD_SINGLEHOST_GROUP)
PREFILL_CONFIG, DECODE_CONFIG = pd_singlehost_role_configs(_CONFIG)
