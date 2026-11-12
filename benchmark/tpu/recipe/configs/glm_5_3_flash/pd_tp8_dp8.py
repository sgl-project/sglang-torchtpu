"""GLM-5.3-Flash Raiden P/D, P: TP8 -> D: DP8: pd.py with the decode's
attention data-parallel, eight workers of attention TP 1 on its host.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/pd_tp8_dp8.py:PREFILL_CONFIG
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/pd_tp8_dp8.py:DECODE_CONFIG

Raiden moves a request as a pool transfer (SGLANG_RAIDEN_PIPELINE_TRANSFER on
both engines): the prefill's ranks, which each hold every sparse-MLA page,
stripe the pages between them, and the decode worker gathers their head shards
of the KDA state into its full-width rows.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import PD_SINGLEHOST_GROUP
from benchmark.tpu.recipe.configs.glm_5_3_flash.pd import roles

GROUP = deepcopy(PD_SINGLEHOST_GROUP)
PREFILL_CONFIG, DECODE_CONFIG = roles(decode_dp=True)
