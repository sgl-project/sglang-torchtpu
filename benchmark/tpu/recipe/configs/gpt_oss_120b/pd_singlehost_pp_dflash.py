"""Raiden P/D: a PP8 DFlash prefill feeding a TP8/EP8 decode, experimental spec worker."""

import os

from benchmark.tpu.recipe.configs.gpt_oss_120b.pd_singlehost_pp_spec import (
    pp_spec_role_configs,
)

DRAFT_MODEL = (
    os.environ.get("GPT_OSS_DRAFT_MODEL")
    or os.environ.get("DRAFT_MODEL")
    or "z-lab/gpt-oss-120b-DFlash"
)
# Fixed by the z-lab checkpoint, as in the single-host DFlash perf test.
DFLASH_BLOCK_SIZE = 3

GROUP, PREFILL_CONFIG, DECODE_CONFIG = pp_spec_role_configs(
    spec_args=dict(
        speculative_algorithm="DFLASH",
        speculative_draft_model_path=DRAFT_MODEL,
        speculative_dflash_block_size=DFLASH_BLOCK_SIZE,
    ),
    # Broadcast only the sites that can differ across the decode's TP ranks.
    spec_envs={"SGLANG_SPEC_TP_SYNC": "rng"},
    # DFlash's auto min-free-slots delay holds decode rows the batch bucket pays for
    # anyway; 1 disables it. PP rejects the flag, so the prefill leaves it unset.
    decode_args={"min_free_slots_delay": 1},
)
