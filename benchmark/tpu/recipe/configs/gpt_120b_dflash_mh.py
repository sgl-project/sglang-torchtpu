"""TPU serving configuration. Callers copy CONFIG before overriding it."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b.multihost_base import (
    CONFIG as BASE_CONFIG,
)

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "max_total_tokens": 131072,
        "speculative_algorithm": "DFLASH",
        "speculative_draft_model_path": "z-lab/gpt-oss-120b-DFlash",
        "speculative_dflash_block_size": 6,
    }
)
CONFIG["runtime_env"].update({"SGLANG_SPEC_TP_SYNC": "rng"})
