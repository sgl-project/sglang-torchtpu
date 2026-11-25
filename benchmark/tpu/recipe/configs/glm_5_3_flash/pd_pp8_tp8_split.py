"""GLM-5.3-Flash Raiden P/D, P: PP8 -> D: TP8 on a split cache: pd_pp8_tp8.py
with the decode's latent and index-key caches split eight ways (agentx.py's
c32 split), so each stage sends each decode rank its eighth of every page.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/pd_pp8_tp8_split.py:PREFILL_CONFIG
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/pd_pp8_tp8_split.py:DECODE_CONFIG
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash import pd_pp8_tp8
from benchmark.tpu.recipe.configs.glm_5_3_flash.agentx import SPARSE_CORE_FIX

GROUP = deepcopy(pd_pp8_tp8.GROUP)
PREFILL_CONFIG = deepcopy(pd_pp8_tp8.PREFILL_CONFIG)
DECODE_CONFIG = deepcopy(pd_pp8_tp8.DECODE_CONFIG)
_env = DECODE_CONFIG["runtime_env"]
_env.update(
    {
        "SGLANG_TPU_DSA_MLA_KV_SHARDS": "8",
        # Each selected cache row is gathered by the rank that holds it.
        "SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT": "1",
        "SGLANG_TPU_DSA_INDEX_KV_SHARD": "1",
        "SGLANG_TPU_DSA_KV_SHARD_RAGGED_BLOCK": "128",  # decode: own candidates only
    }
)
_env["LIBTPU_INIT_ARGS"] += SPARSE_CORE_FIX
