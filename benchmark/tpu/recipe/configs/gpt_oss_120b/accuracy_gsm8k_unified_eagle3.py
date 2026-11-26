"""GPT-OSS-120B full GSM8K at low effort on the unified Full/SWA KV pool with EAGLE3.

accuracy_gsm8k.py owns the backbone serving shape the nightly score comes from;
this file only adds the draft model and the unified pool, so the two scores are
directly comparable.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/gpt_oss_120b/accuracy_gsm8k_unified_eagle3.py
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b.accuracy_gsm8k import (
    CONFIG as GSM8K_CONFIG,
)

DRAFT_MODEL_PATH = os.environ.get("DRAFT_MODEL", "lmsys/EAGLE3-gpt-oss-120b-bf16")

CONFIG = deepcopy(GSM8K_CONFIG)
CONFIG["server_args"].update(
    {
        "enable_unified_memory": True,
        # Same KV bytes as the backbone's 262144: a separate pool of N tokens
        # also holds 0.8 N SWA tokens, the unified pool shares 1.8 N slots.
        "max_total_tokens": 471040,
        "speculative_algorithm": "EAGLE3",
        "speculative_draft_model_path": DRAFT_MODEL_PATH,
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
    }
)
