"""GPT-OSS-120B with DFlash speculative decoding, single-host gate.

server.py owns how this model is served; this file changes only what the
gate needs.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_120b/dflash.py
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b.server import CONFIG as SERVER_CONFIG

CONCURRENCY = 16
CONTEXT_LENGTH = 8192
PREFILL_TOKENS = 4096
DRAFT_MODEL_PATH = os.environ.get("DRAFT_MODEL", "z-lab/gpt-oss-120b-DFlash")

CONFIG = deepcopy(SERVER_CONFIG)
CONFIG["server_args"].update(
    {
        # The gate resolves either side from MODEL/DRAFT_MODEL so it also runs on
        # a host that has the checkpoints in the HF cache rather than /weights.
        "model_path": os.environ.get("MODEL", "/weights/gpt-oss-120b"),
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "tpu_compile_bs": [1, 2, 4, CONCURRENCY],
        "tpu_compile_tokens": [512, 2048, PREFILL_TOKENS],
        "context_length": CONTEXT_LENGTH,
        "max_total_tokens": 8192,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "speculative_algorithm": "DFLASH",
        "speculative_draft_model_path": DRAFT_MODEL_PATH,
        "speculative_dflash_block_size": 6,
        "reasoning_parser": "gpt-oss",
        "served_model_name": "gpt-oss-120b",
    }
)
CONFIG["runtime_env"].update(
    {
        # Broadcast only the sites that can differ across ranks (memory init and
        # the RNG draws); the deterministic ones cost a collective per step.
        "SGLANG_SPEC_TP_SYNC": "rng",
        # Route DFLASH to sglang_tpu.experimental.TpuBlockSpecWorker, the worker the
        # DFlash gates are measured on (#1413).
        "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER": "1",
    }
)
