"""GPT-OSS-120B Raiden P/D: one TP8/EP8 host for each engine.

Server tuning is defined here. Only model locations, service ports, and readiness
budgets come from the deployment environment. Supply TPU runtime/cache/profiling
overrides through the launcher's runtime_envs dictionary.
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

GROUP = {"hosts": 2, "prefill_engines": 1, "decode_engines": 1}

_COMMON_CONFIG = {
    "server_args": {
        **deepcopy(_FP8_BASE_CONFIG["server_args"]),
        "model_path": (
            os.environ.get("GPT_OSS_MODEL")
            or os.environ.get("MODEL")
            or "/weights/gpt-oss-120b"
        ),
        "served_model_name": "gpt-oss-120b",
        "host": "0.0.0.0",
        "tp_size": 8,
        "ep_size": 8,
        "dp_size": 1,
        "enable_dp_attention": False,
        "enable_dp_lm_head": False,
        "page_size": 64,
        # Use checkpoint-provided MXFP4 weights, with FP8 KV and radix caching.
        "quantization": None,
        "kv_cache_dtype": "fp8_e4m3",
        "disable_radix_cache": False,
        "mem_fraction_static": 0.35,
        "max_total_tokens": 393216,
        "max_running_requests": 64,
        "chunked_prefill_size": 4096,
        "max_prefill_tokens": 4096,
        "context_length": 9236,
        "enable_torch_compile": True,
        "torch_compile_max_bs": 64,
        "reasoning_parser": "gpt-oss",
        "sampling_backend": "pytorch",
        "skip_server_warmup": True,
        "disaggregation_transfer_backend": "raiden",
    },
    "runtime_envs": {
        **_FP8_BASE_CONFIG["runtime_env"],
        # GPT-OSS does not support the SparseCore MoE gather path.
        "SGLANG_TPU_FUSED_MOE_SC_GATHER": "0",
        "SGLANG_RAIDEN_KV_PORT": os.environ.get("SGLANG_RAIDEN_KV_PORT", "31000"),
        "LB_PORT": os.environ.get("LB_PORT", "8000"),
        "HEALTH_TIMEOUT": os.environ.get("WAIT_SECS", "3600"),
        # Preserve GPT_OSS_MODEL precedence over the generic MODEL override.
        "MODEL": None,
    },
}

# Independent role configs: mutating one engine must not change the other.
PREFILL_CONFIG = deepcopy(_COMMON_CONFIG)
PREFILL_CONFIG["server_args"].update(
    port=int(os.environ.get("PREFILL_PORT", "30000")),
    disaggregation_mode="prefill",
    disaggregation_bootstrap_port=int(os.environ.get("BOOTSTRAP_PORT", "8998")),
)

DECODE_CONFIG = deepcopy(_COMMON_CONFIG)
DECODE_CONFIG["server_args"].update(
    port=int(os.environ.get("DECODE_PORT", "30000")),
    disaggregation_mode="decode",
    hicache_mem_layout="page_first_fused",
    hicache_io_backend="raiden",
)
