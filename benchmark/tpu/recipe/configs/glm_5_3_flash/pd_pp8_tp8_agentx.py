"""GLM-5.3-Flash AgentX P/D, P: PP8 -> D: TP8 on a split cache, at one
concurrency (AGENTX_PROFILE: 1, 8, 16 or 32 lanes; default 32).

The decode runs agentx.py's profile for that concurrency (running requests,
decode buckets, MTP steps and their schedule, kernels) without its prefix
cache and host tier, which a P/D decode does not keep. Its cache is split
eight ways with the index-key split at every concurrency, the layout the pool
transfer was validated with (pd_pp8_tp8_split.py), and has no raw ring, which
needs a newer tpu-sync. The prefill is pd_pp8_tp8.py's PP8 at 1M context with
the prefix-cache settings of the AgentX recipes (a KDA checkpoint at every
prefill chunk, none during decode); the AgentX recipe turns on the radix cache
and the host tier.
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.glm_5_3_flash import pd_pp8_tp8_split
from benchmark.tpu.recipe.configs.glm_5_3_flash.agentx import config_for

CONCURRENCY = int(os.environ.get("AGENTX_PROFILE") or 32)

GROUP = deepcopy(pd_pp8_tp8_split.GROUP)
PREFILL_CONFIG = deepcopy(pd_pp8_tp8_split.PREFILL_CONFIG)
DECODE_CONFIG = deepcopy(pd_pp8_tp8_split.DECODE_CONFIG)

_agentx = config_for(CONCURRENCY)
_args, _env = _agentx["server_args"], _agentx["runtime_env"]
CONTEXT_LENGTH = _args["context_length"]

# Decode: the profile's scheduling, compile grid, MTP and kernels.
DECODE_CONFIG["server_args"].update(
    {
        key: _args[key]
        for key in (
            "context_length",
            "max_running_requests",
            "torch_compile_max_bs",
            "tpu_compile_bs",
            "tpu_compile_tokens",
            "chunked_prefill_size",
            "max_prefill_tokens",
            "tpu_compile_tracer",
            "speculative_num_steps",
            "speculative_num_draft_tokens",
            "scheduler_recv_interval",
            "enable_metrics",
            "enable_request_time_stats_logging",
        )
    }
)
# Only some profiles pin the decode-time KDA checkpoint interval.
if "mamba_track_interval" in _args:
    DECODE_CONFIG["server_args"]["mamba_track_interval"] = _args["mamba_track_interval"]
# The profile's kernels and MTP schedule, on the validated split cache.
_split_env = pd_pp8_tp8_split.DECODE_CONFIG["runtime_env"]
_layout = (
    "LIBTPU_INIT_ARGS",
    "SGLANG_TPU_DSA_MLA_KV_SHARDS",
    "SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT",
    "SGLANG_TPU_DSA_INDEX_KV_SHARD",
    "SGLANG_TPU_DSA_KV_SHARD_RAGGED_BLOCK",
)
DECODE_CONFIG["runtime_env"].update(
    {
        key: value
        for key, value in _env.items()
        if key
        not in (
            *_layout,
            "SGLANG_TPU_HICACHE_STATE_RATIO",
            "SGLANG_TPU_DSA_KPOOL_RAW_RING",
        )
    }
)
assert all(DECODE_CONFIG["runtime_env"][key] == _split_env[key] for key in _layout)

# Prefill: PP8 at the profile's context, with the AgentX prefix-cache settings.
PREFILL_CONFIG["server_args"]["context_length"] = CONTEXT_LENGTH
PREFILL_CONFIG["server_args"].update(
    {
        "hybrid_cache_checkpoints_level": 3,
        # No KDA checkpoints during decode: a prefill engine never reuses them.
        "mamba_track_interval": CONTEXT_LENGTH,
    }
)
