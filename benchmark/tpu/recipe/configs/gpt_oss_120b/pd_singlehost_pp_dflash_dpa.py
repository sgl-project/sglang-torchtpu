"""Raiden P/D: a PP8 DFlash prefill feeding a TP8/EP8 decode with DP attention, experimental spec worker."""

from benchmark.tpu.recipe.configs.gpt_oss_120b.pd_singlehost_pp_dflash import (
    DRAFT_MODEL,
)
from benchmark.tpu.recipe.configs.gpt_oss_120b.pd_singlehost_pp_spec import (
    pp_spec_role_configs,
)

# DP8 was the fastest decode DP size for 8k1k c64 (DP1/2/4/8 swept).
DECODE_DP = 8

GROUP, PREFILL_CONFIG, DECODE_CONFIG = pp_spec_role_configs(
    # Block size comes from the draft config.
    spec_args=dict(
        speculative_algorithm="DFLASH",
        speculative_draft_model_path=DRAFT_MODEL,
    ),
    spec_envs={"SGLANG_SPEC_TP_SYNC": "rng"},
    # As in pd_singlehost_pp_dflash.py; PP rejects the flag, so the prefill leaves it unset.
    decode_args={"min_free_slots_delay": 1},
    decode_dp=DECODE_DP,
)
