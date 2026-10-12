"""GPT-OSS-20B (MXFP4 MoE) on one tpu7x host as four pipeline stages, each a
two-device prefill context-parallel group (PP4 x PCP2).

Needs a torch_tpu with standalone process subgroups.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_20b/pp4_pcp2.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONCURRENCY = 32
PREFILL_TOKENS = 4096

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "openai/gpt-oss-20b",
        "served_model_name": "gpt-oss-20b",
        "pp_size": 4,
        "tp_size": 2,
        "pp_max_micro_batch_size": 4,
        # Each stage shards a prefill over its two devices and keeps every
        # expert on both, so each device routes only its own tokens.
        "attn_cp_size": 2,
        "enable_prefill_cp": True,
        "cp_strategy": "zigzag",
        "moe_dp_size": 2,
        # The checkpoint's own mxfp4 config; fp8 would re-quantize it.
        "quantization": None,
        "attention_backend": "pallas_tpu_rpa_batched",
        "page_size": 256,
        "kv_cache_dtype": "bfloat16",
        "context_length": 16384,
        "max_total_tokens": 262144,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "tpu_compile_tokens": [512, 1024, 2048, PREFILL_TOKENS],
        "tpu_compile_bs": [8, 16, CONCURRENCY],
        "mem_fraction_static": 0.8,
        "tpu_compile_tracer": "dynamo",
        # A stage waiting for the previous one looks idle to the watchdog.
        "watchdog_timeout": 3600,
    }
)
# SparseCore ragged-gather MoE cannot be lowered for GPT-OSS (see gpt_oss_120b/server.py).
CONFIG["runtime_env"].update({"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0"})
