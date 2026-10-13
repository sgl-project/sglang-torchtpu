"""GPT-OSS-20B (MXFP4 MoE) on one tpu7x host with prefill context parallelism
(tp8, PCP8, MoE DP8). The PR gate's PCP configuration; qwen3_30b_a3b_fp8_pcp
keeps the 30B checkpoint on the nightly.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_20b/pcp.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONCURRENCY = 32
PREFILL_TOKENS = 8192

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "openai/gpt-oss-20b",
        "served_model_name": "gpt-oss-20b",
        "tp_size": 8,
        # Attention CP 8 shards each prefill over the TP group; MoE DP 8 keeps
        # every expert on every core, so each core routes only its own tokens.
        "attn_cp_size": 8,
        "enable_prefill_cp": True,
        "cp_strategy": "zigzag",
        "moe_dp_size": 8,
        "moe_runner_backend": "pallas_tpu",
        "sampling_backend": "pytorch",
        # The checkpoint's own mxfp4 config; fp8 would re-quantize it.
        "quantization": None,
        # Batched RPA implements the attention sinks the model needs.
        "attention_backend": "pallas_tpu_rpa_batched",
        "page_size": 256,
        "context_length": PREFILL_TOKENS,
        "max_total_tokens": 131072,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        # CP batches bucket their local rows (1/8 of the tokens): 512 and 1024
        # hold the needles, 2048 a full chunk; 8192 covers non-CP batches.
        "tpu_compile_tokens": [512, 1024, 2048, PREFILL_TOKENS],
        "tpu_compile_bs": [8, 16, CONCURRENCY],
        "mem_fraction_static": 0.6,
        # The needle probe resends shared prefixes.
        "disable_radix_cache": False,
        "reasoning_parser": "gpt-oss",
        "watchdog_timeout": 3600,
    }
)
# SparseCore ragged-gather MoE cannot be lowered for GPT-OSS (see gpt_oss_120b/server.py).
CONFIG["runtime_env"].update({"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0"})
