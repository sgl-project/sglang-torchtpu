"""Raiden P/D with one tpu7x host per engine: a PP8 prefill (eight single-device
stages) feeding a TP8 decode, Qwen3.8-Flash-Next-FP8 text only.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/qwen3_8_flash_next/pd_singlehost_pp8.py:PREFILL_CONFIG
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import (
    PD_SINGLEHOST_GROUP,
    pd_singlehost_role_configs,
)
from benchmark.tpu.recipe.configs.qwen3_8_flash_next import server_pp8, server_tp8

GROUP = deepcopy(PD_SINGLEHOST_GROUP)
PREFILL_CONFIG, _ = pd_singlehost_role_configs(server_pp8.CONFIG)
_, DECODE_CONFIG = pd_singlehost_role_configs(server_tp8.CONFIG)
# A non-PP decode names its layer pools to pull from a PP producer.
DECODE_CONFIG["runtime_env"]["SGLANG_RAIDEN_PIPELINE_TRANSFER"] = "1"

# TP8 prefill -> TP8 decode on the non-pipeline Raiden path: proves the model's
# P/D state (GDN + PLE) before the pipeline transfer carries it.
TP_PREFILL_CONFIG, TP_DECODE_CONFIG = pd_singlehost_role_configs(server_tp8.CONFIG)
