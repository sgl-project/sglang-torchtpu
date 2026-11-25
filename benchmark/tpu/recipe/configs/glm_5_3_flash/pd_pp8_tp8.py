"""GLM-5.3-Flash Raiden P/D, P: PP8 -> D: TP8: pd.py with the prefill as a
pipeline of eight stages (server_pp8.py), the MTP draft on its last stage.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/pd_pp8_tp8.py:PREFILL_CONFIG
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/pd_pp8_tp8.py:DECODE_CONFIG

Raiden moves a request as a pool transfer (SGLANG_RAIDEN_PIPELINE_TRANSFER on
the decode): each stage sends every sparse-MLA page of its layers to every
decode rank and cuts its KDA state rows into the ranks' head shards; the last
stage also sends the draft's pages and the aux row.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import PD_SINGLEHOST_GROUP
from benchmark.tpu.recipe.configs.glm_5_3_flash.pd import roles
from benchmark.tpu.recipe.configs.glm_5_3_flash.server_pp8 import CONFIG as PP8_CONFIG

CONTEXT_LENGTH = 131072

GROUP = deepcopy(PD_SINGLEHOST_GROUP)
PREFILL_CONFIG, DECODE_CONFIG = roles()
_pp_args, _pp_env = PP8_CONFIG["server_args"], PP8_CONFIG["runtime_env"]
PREFILL_CONFIG["server_args"].update(
    {
        key: _pp_args[key]
        for key in (
            "tp_size",
            "pp_size",
            "ep_size",
            "enable_attn_tp_input_scattered",
            "disable_overlap_schedule",
            "tpu_compile_tracer",
            "dist_timeout",
            "chunked_prefill_size",
            "max_prefill_tokens",
            "tpu_compile_tokens",
        )
    }
)
PREFILL_CONFIG["runtime_env"].update(
    {
        key: value
        for key, value in _pp_env.items()
        if key not in PREFILL_CONFIG["runtime_env"]
    }
)
for _role in (PREFILL_CONFIG, DECODE_CONFIG):
    _role["server_args"]["context_length"] = CONTEXT_LENGTH
    _role["runtime_env"]["SGLANG_RAIDEN_PIPELINE_TRANSFER"] = "1"
