"""GPT-OSS-120B DP attention with one request per group, single-host gate.

server.py owns how this model is served; this file changes only what the
gate needs.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_120b/dpa_idle_group.py
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b.server import CONFIG as SERVER_CONFIG

CONCURRENCY = 2
CONTEXT_LENGTH = 8192
PREFILL_TOKENS = 4096
DP_SIZE = 2

CONFIG = deepcopy(SERVER_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "/weights/gpt-oss-120b",
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "tpu_compile_tokens": [512, 1024, 2048, PREFILL_TOKENS],
        "context_length": CONTEXT_LENGTH,
        "max_total_tokens": 8192,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        # tp8 / dp2 gives attention_tp = 4.
        "dp_size": DP_SIZE,
        "enable_dp_attention": True,
        # attention_tp > 1, which TPU requires the LM head to follow.
        "enable_dp_lm_head": True,
        "reasoning_parser": "gpt-oss",
        "served_model_name": "gpt-oss-120b",
    }
)
