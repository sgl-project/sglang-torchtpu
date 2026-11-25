"""GLM-5.3-Flash Raiden P/D with the MTP head on decode: one tpu7x host
(tp8) per engine.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/pd.py:PREFILL_CONFIG   # prefill host
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/pd.py:DECODE_CONFIG    # decode host

and front them with the P/D router (``sglang_router.launch_router
--pd-disaggregation --mini-lb``). At equal attention TP, Raiden pulls each
request's sparse-MLA pages (target and MTP layers) and its KDA conv /
recurrent state; the aux blob carries the draft's hidden seed and, with MTP
index sharing (mtp.py's default; GLM_FLASH_MTP_INDEX_SHARE=0 turns it off on
both engines), the draft's DSA top-k at the last prompt row.

roles() also builds either engine with attention DP: eight workers of
attention TP 1 on the same host.

Neither engine keeps a prefix cache.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.base import (
    PD_SINGLEHOST_GROUP,
    pd_singlehost_role_configs,
)
from benchmark.tpu.recipe.configs.glm_5_3_flash.mtp import CONFIG as MTP_CONFIG
from benchmark.tpu.recipe.configs.glm_5_3_flash.server import HICACHE_ARGS

_SERVER = deepcopy(MTP_CONFIG)
_args = _SERVER["server_args"]
for _key in HICACHE_ARGS:
    _args.pop(_key, None)
# No prefix cache either: the P/D decode never keeps one, and on the prefill the
# KDA snapshot slots (extra_buffer) would cap running requests at a fifth of
# max_mamba_cache_size.
_args["disable_radix_cache"] = True
_args.pop("mamba_radix_cache_strategy", None)
_args.update(
    {
        # The sampler every TPU P/D gate pins; greedy picks the argmax either way.
        "sampling_backend": "pytorch",
        # A disaggregated engine's warmup request needs its peer and the router.
        "skip_server_warmup": True,
    }
)
# Log any XLA compile while serving.
_SERVER["runtime_env"]["SGLANG_TPU_CHECK_RECOMPILATION"] = "1"

GROUP = deepcopy(PD_SINGLEHOST_GROUP)
# Attention DP over the host's eight chips. The MTP draft shares the target's
# lm_head and runs in a worker's attention-TP group, so the head follows it.
_ATTN_DP = {"dp_size": 8, "enable_dp_attention": True, "enable_dp_lm_head": True}


def roles(*, prefill_dp: bool = False, decode_dp: bool = False):
    """(PREFILL_CONFIG, DECODE_CONFIG), each tp8 or with attention DP."""
    prefill, decode = pd_singlehost_role_configs(_SERVER)
    # The split MLA pools (target and draft) and the KDA state pool have no host
    # copy, so decode retraction cannot back a request up; it aborts it instead.
    decode["server_args"]["disaggregation_decode_retraction_backup"] = "none"
    for role, attn_dp in ((prefill, prefill_dp), (decode, decode_dp)):
        if attn_dp:
            role["server_args"].update(_ATTN_DP)
    if prefill_dp != decode_dp:
        # Attention TP differs, so the request moves as a pool transfer.
        for role in (prefill, decode):
            role["runtime_env"]["SGLANG_RAIDEN_PIPELINE_TRANSFER"] = "1"
    return prefill, decode


PREFILL_CONFIG, DECODE_CONFIG = roles()
