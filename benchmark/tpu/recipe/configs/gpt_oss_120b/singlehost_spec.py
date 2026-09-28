"""GPT-OSS-120B EAGLE3 configuration for the single-host GSM8K quality gate."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b.server import CONFIG as BASE_CONFIG

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "context_length": 8192,
        "max_running_requests": 16,
        "torch_compile_max_bs": 16,
        "chunked_prefill_size": 4096,
        "max_prefill_tokens": 4096,
        "max_total_tokens": 8192,
        "model_path": "/weights/gpt-oss-120b",
        "tpu_compile_tokens": [512, 2048, 4096],
        "tpu_compile_bs": [1, 2, 4, 16],
        "reasoning_parser": "gpt-oss",
        "served_model_name": "gpt-oss-120b",
        "speculative_algorithm": "EAGLE3",
        "speculative_draft_model_path": "lmsys/EAGLE3-gpt-oss-120b-bf16",
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
    }
)
