"""Qwen3-Coder-30B (FP8 MoE) on one tpu7x host, tp8/ep8. Callers copy CONFIG
before overriding it.

Sample usage (from the repo root):
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/qwen3_coder_30b_a3b_fp8.py
    # add --dry-run to print the resolved sglang.launch_server command
    # (inherited base args included) without launching.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG

CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update(
    {
        # HF repo id, not a /weights path: this config predates the shared
        # model-weights-rox disk and pulls from the HF cache.
        "model_path": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        "tp_size": 8,
        "ep_size": 8,
        # Sized for the lm-eval MMLU sweep at max_running_requests=64.
        "context_length": 8192,
        "max_total_tokens": 262144,
        # Leaves room for that KV pool at tp=8.
        "mem_fraction_static": 0.3,
    }
)
