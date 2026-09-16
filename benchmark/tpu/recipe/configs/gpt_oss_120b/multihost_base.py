"""Shared base for the two-host gpt-oss-120b spec configs (dflash, eagle3).

Holds the model identity and the two-host tp=16/ep=16 layout both variants
launch with; each variant overrides its draft algorithm and KV budget. Callers
copy CONFIG before overriding it.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/gpt-oss-120b",
        "served_model_name": "gpt-oss-120b",
        "reasoning_parser": "gpt-oss",
        "nnodes": 2,
        "tp_size": 16,
        "ep_size": 16,
        "context_length": 8192,
        "page_size": 64,
        "mem_fraction_static": 0.6,
        "quantization": None,
        "enable_metrics": True,
        "tpu_compile_tokens": [512, 1024, 2048, 4096],
        "tpu_compile_bs": [1, 2, 4, 8, 16],
        "tpu_compile_tracer": "dynamo",
    }
)
CONFIG["runtime_env"].update({"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0"})
