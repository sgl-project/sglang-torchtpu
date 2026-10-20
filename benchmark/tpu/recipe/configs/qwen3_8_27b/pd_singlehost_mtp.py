"""Qwen3.8-27B MTP (3/1/4) on the experimental TPU spec worker, Raiden P/D with
one tpu7x host per engine, sized like mtp_gate.py.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/qwen3_8_27b/pd_singlehost_mtp.py:PREFILL_CONFIG
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import (
    PD_SINGLEHOST_GROUP,
    pd_singlehost_role_configs,
)
from benchmark.tpu.recipe.configs.qwen3_8_27b import mtp_gate

_CONFIG = deepcopy(mtp_gate.CONFIG)
# Source line of any serving-time compile, for the CI log.
_CONFIG["runtime_env"]["SGLANG_TPU_CHECK_RECOMPILATION_SITES"] = "1"

GROUP = deepcopy(PD_SINGLEHOST_GROUP)
PREFILL_CONFIG, DECODE_CONFIG = pd_singlehost_role_configs(_CONFIG)
