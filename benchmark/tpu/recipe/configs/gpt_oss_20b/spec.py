"""GPT-OSS-20B with EAGLE3 speculative decoding, for the single-host PR gate.

server.py owns how the model is served; this file adds the draft model and the
small shape the gate drives.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_20b/spec.py
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_20b.server import CONFIG as SERVER_CONFIG

CONCURRENCY = 4
CONTEXT_LENGTH = 4096
PREFILL_TOKENS = 2048
# The gate resolves both against the CI weights mount and passes them in; the
# ids are what a reader reproduces with.
MODEL_PATH = os.environ.get("MODEL", "openai/gpt-oss-20b")
DRAFT_MODEL_PATH = os.environ.get("DRAFT_MODEL", "zhuyksir/EAGLE3-gpt-oss-20b-bf16")

CONFIG = deepcopy(SERVER_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": MODEL_PATH,
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "tpu_compile_bs": [1, 2, CONCURRENCY],
        "tpu_compile_tokens": [512, 1024, PREFILL_TOKENS],
        "context_length": CONTEXT_LENGTH,
        "max_total_tokens": 2 * CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "speculative_algorithm": "EAGLE3",
        "speculative_draft_model_path": DRAFT_MODEL_PATH,
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
        # The EAGLE3 draft checkpoint carries its own modeling code.
        "trust_remote_code": True,
        "reasoning_parser": "gpt-oss",
    }
)
