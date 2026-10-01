"""GPT-OSS-20B (MXFP4 MoE) on one tpu7x host, 4 chips x 2 cores = tp8, sized for
the high-reasoning GPQA / AIME25 / MMLU nightly at concurrency 128.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_20b/accuracy.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONCURRENCY = 128
CONTEXT_LENGTH = 131072
PREFILL_TOKENS = 4096

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "openai/gpt-oss-20b",
        "served_model_name": "gpt-oss-20b",
        "tp_size": 8,
        # EP for the experts, as in gpt_oss_120b/server.py.
        "ep_size": 8,
        # The checkpoint's own mxfp4 config; fp8 would re-quantize it.
        "quantization": None,
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "tpu_compile_bs": [1, 8, 32, 64, CONCURRENCY],
        "tpu_compile_tokens": [PREFILL_TOKENS],
        "context_length": CONTEXT_LENGTH,
        "max_total_tokens": 64 * CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "mem_fraction_static": 0.8,
        "disable_overlap_schedule": True,
        "reasoning_parser": "gpt-oss",
        "tpu_compile_tracer": "dynamo",
        # The evaluator asks for the full 131072-token budget before counting
        # prompt tokens; cap generation to the remaining context instead.
        "allow_auto_truncate": True,
    }
)
# SparseCore ragged-gather MoE cannot be lowered for GPT-OSS (see gpt_oss_120b/server.py).
CONFIG["runtime_env"].update({"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0"})
