"""TPU serving configuration. Callers copy CONFIG before overriding it."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/gpt-oss-120b",
        "nnodes": 2,
        "tp_size": 16,
        "ep_size": 16,
        "context_length": 8192,
        "max_total_tokens": 131072,
        "page_size": 64,
        "mem_fraction_static": 0.6,
        "quantization": None,
        "enable_metrics": True,
        "tpu_compile_tokens": [512, 1024, 2048, 4096],
        "tpu_compile_bs": [1, 2, 4, 8, 16],
        "reasoning_parser": "gpt-oss",
        "served_model_name": "gpt-oss-120b",
        "speculative_algorithm": "DFLASH",
        "speculative_draft_model_path": "z-lab/gpt-oss-120b-DFlash",
        "tpu_compile_tracer": "dynamo",
        "speculative_dflash_block_size": 6,
    }
)
CONFIG["runtime_env"].update(
    {"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0", "SGLANG_SPEC_TP_SYNC": "rng"}
)
