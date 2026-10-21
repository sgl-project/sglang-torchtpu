"""Muse Spark for one agentic session per host: decode megakernel plus DFlash.

The AgentX cc=1 configuration: one running request over long cached prefixes
(up to 256K tokens), DFlash with 8 draft tokens, and the whole-decoder
megakernel for decode and verify steps.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/muse_spark/agentic.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.muse_spark.server import CONFIG as SERVER_CONFIG

CONFIG = deepcopy(SERVER_CONFIG)
CONFIG["server_args"].update(
    {
        "max_running_requests": 1,
        "torch_compile_max_bs": 1,
        "tpu_compile_bs": [1],
        # The pool size the agentic runs were tuned with; packed KV heads (below)
        # halve the per-token KV bytes.
        "max_total_tokens": 917504,
        "speculative_algorithm": "DFLASH",
        "speculative_draft_model_path": "RadixArk/Muse-Spark-Open-DFlash2",
        "speculative_num_draft_tokens": 8,
        "scheduler_recv_interval": 8,
        # Up to 128 decode steps run between the prefill chunks of a new turn.
        "prefill_decode_interval": 128,
        "enable_cache_report": True,
        "enable_metrics": True,
    }
)
CONFIG["runtime_env"].update(
    {
        "SGLANG_TPU_MUSE_MEGAKERNEL": "1",
        "SGLANG_TPU_KV_PACK_HEADS": "1",
        "SGLANG_DFLASH_ROW_PARALLEL_FC": "1",
        "SGLANG_DFLASH_CONV_MATMUL_EXPAND": "1",
        "SGLANG_TPU_DFLASH_FP8_DRAFT": "1",
        "SGLANG_SPEC_TP_SYNC": "rng",
        "SGLANG_TPU_ALLOW_SPEC_RECOMPILE": "1",
    }
)
