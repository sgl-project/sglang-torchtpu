"""GLM-5.3-Flash Raiden P/D, P: DP8 -> D: DP8: pd.py with both engines'
attention data-parallel, eight workers of attention TP 1 on each host.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/pd_dp8_dp8.py:PREFILL_CONFIG
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/pd_dp8_dp8.py:DECODE_CONFIG

A decode worker's attention TP equals its prefill worker's, so Raiden pulls a
request as pd.py does.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import PD_SINGLEHOST_GROUP
from benchmark.tpu.recipe.configs.glm_5_3_flash.pd import roles

GROUP = deepcopy(PD_SINGLEHOST_GROUP)
PREFILL_CONFIG, DECODE_CONFIG = roles(prefill_dp=True, decode_dp=True)
