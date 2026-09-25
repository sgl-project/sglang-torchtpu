"""One-host-per-engine Raiden P/D with decode-side radix cache reuse."""

from copy import deepcopy

from benchmark.tpu.recipe.configs.gpt_oss_120b import pd_singlehost

GROUP = deepcopy(pd_singlehost.GROUP)
PREFILL_CONFIG = deepcopy(pd_singlehost.PREFILL_CONFIG)
DECODE_CONFIG = deepcopy(pd_singlehost.DECODE_CONFIG)
DECODE_CONFIG["server_args"]["disaggregation_decode_enable_radix_cache"] = True
DECODE_CONFIG["runtime_envs"]["SGLANG_ENABLE_UNIFIED_RADIX_TREE"] = "1"
