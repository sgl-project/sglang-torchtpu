"""Raiden P/D: a PP8 EAGLE3 prefill feeding a TP8/EP8 decode, experimental spec worker."""

import os

from benchmark.tpu.recipe.configs.gpt_oss_120b.pd_singlehost_pp_spec import (
    pp_spec_role_configs,
)

DRAFT_MODEL = (
    os.environ.get("GPT_OSS_DRAFT_MODEL")
    or os.environ.get("DRAFT_MODEL")
    or "lmsys/EAGLE3-gpt-oss-120b-bf16"
)

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
)
