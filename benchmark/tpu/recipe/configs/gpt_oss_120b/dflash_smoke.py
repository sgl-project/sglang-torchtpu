"""GPT-OSS-120B DFlash bring-up smoke, single-host gate.

server.py owns how this model is served; this file changes only what the
smoke needs.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_120b/dflash_smoke.py
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b.server import CONFIG as SERVER_CONFIG

CONCURRENCY = 4
CONTEXT_LENGTH = 4096
PREFILL_TOKENS = 2048

CONFIG = deepcopy(SERVER_CONFIG)
CONFIG["server_args"].update(
    {
        # The gate resolves either side from MODEL/DRAFT_MODEL so it also runs
        # on a host that has the checkpoints in the HF cache rather than /weights.
        "model_path": os.environ.get("MODEL", "/weights/gpt-oss-120b"),
        "speculative_draft_model_path": os.environ.get(
            "DRAFT_MODEL", "z-lab/gpt-oss-120b-DFlash"
        ),
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "tpu_compile_bs": [1, 2, CONCURRENCY],
        "tpu_compile_tokens": [512, 1024, PREFILL_TOKENS],
        "context_length": CONTEXT_LENGTH,
        "max_total_tokens": 8192,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "speculative_algorithm": "DFLASH",
        "speculative_dflash_block_size": 6,
        "reasoning_parser": "gpt-oss",
        "served_model_name": "gpt-oss-120b",
    }
)
# Route DFLASH to sglang_tpu.experimental.TpuBlockSpecWorker, the worker the
# DFlash gates are measured on (#1413).
CONFIG["runtime_env"].update({"SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER": "1"})
