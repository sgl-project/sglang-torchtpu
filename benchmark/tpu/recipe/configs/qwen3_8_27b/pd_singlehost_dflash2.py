"""Qwen3.8-27B DFlash2 on the experimental TPU spec worker, Raiden P/D with one
tpu7x host per engine, sized like mtp_gate.py.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/qwen3_8_27b/pd_singlehost_dflash2.py:PREFILL_CONFIG
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.base import (
    PD_SINGLEHOST_GROUP,
    pd_singlehost_role_configs,
)
from benchmark.tpu.recipe.configs.qwen3_8_27b import mtp_gate

DRAFT_MODEL = os.environ.get("DRAFT_MODEL") or "/weights/Qwen3.8-27B-DFlash2"

_CONFIG = deepcopy(mtp_gate.CONFIG)
for _name in (
    "speculative_num_steps",
    "speculative_eagle_topk",
    "speculative_num_draft_tokens",
):
    _CONFIG["server_args"].pop(_name)
# DFlash2 is served as DFLASH; the draft's block size sets the verify width.
_CONFIG["server_args"].update(
    speculative_algorithm="DFLASH",
    speculative_draft_model_path=DRAFT_MODEL,
)
_CONFIG["runtime_env"]["SGLANG_TPU_CHECK_RECOMPILATION_SITES"] = "1"

GROUP = deepcopy(PD_SINGLEHOST_GROUP)
PREFILL_CONFIG, DECODE_CONFIG = pd_singlehost_role_configs(_CONFIG)
