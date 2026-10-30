"""Raiden P/D: a PP8 EAGLE3 prefill feeding a TP8/EP8 decode with DP attention, experimental spec worker."""

from benchmark.tpu.recipe.configs.gpt_oss_120b.pd_singlehost_pp_eagle3 import (
    DRAFT_MODEL,
)
from benchmark.tpu.recipe.configs.gpt_oss_120b.pd_singlehost_pp_spec import (
    pp_spec_role_configs,
)

# DP8 was the fastest decode DP size for 8k1k c64 (DP1/2/4/8 swept).
DECODE_DP = 8

GROUP, PREFILL_CONFIG, DECODE_CONFIG = pp_spec_role_configs(
    spec_args=dict(
        speculative_algorithm="EAGLE3",
        speculative_draft_model_path=DRAFT_MODEL,
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
    ),
    spec_envs={},
    decode_args={},
    decode_dp=DECODE_DP,
)
