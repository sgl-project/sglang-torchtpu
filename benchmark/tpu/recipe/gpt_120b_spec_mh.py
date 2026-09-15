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
        "dp_size": 4,
        "context_length": 8192,
        "max_total_tokens": 32768,
        "page_size": 64,
        "mem_fraction_static": 0.6,
        "quantization": None,
        "enable_metrics": True,
        "tpu_compile_tokens": [512, 1024, 2048, 4096],
        "tpu_compile_bs": [1, 2, 4, 8, 16],
        "reasoning_parser": "gpt-oss",
        "served_model_name": "gpt-oss-120b",
        "speculative_algorithm": "EAGLE3",
        "speculative_draft_model_path": "lmsys/EAGLE3-gpt-oss-120b-bf16",
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
        "tpu_compile_tracer": "dynamo",
        "enable_dp_attention": True,
        "enable_dp_lm_head": True,
    }
)
CONFIG["runtime_env"].update({"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0"})
