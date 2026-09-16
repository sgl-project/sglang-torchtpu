"""TPU recipe settings; callers copy dictionaries before overriding them."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.qwen3_coder_480b_agentic_mh import (
    CONFIG as BASE_CONFIG,
)

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "speculative_algorithm": "EAGLE3",
        "speculative_draft_model_path": "/weights/SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-SpecForge-EigenAI",
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
    }
)
CONFIG["runtime_env"].update(
    {
        "SGLANG_SIMULATE_ACC_LEN": "2.5",
        "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
        "SGLANG_SIMULATE_ACC_TOKEN_MODE": "fixed",
    }
)
