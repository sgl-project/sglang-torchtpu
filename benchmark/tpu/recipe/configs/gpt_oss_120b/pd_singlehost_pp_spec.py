"""One-host-per-engine Raiden P/D on the experimental spec worker: a PP8 prefill
feeding a TP8/EP8 decode. pd_singlehost_pp_eagle3.py and pd_singlehost_pp_dflash.py
pick the drafter.
"""

import os
from copy import deepcopy
from typing import Any

from benchmark.tpu.recipe.configs.gpt_oss_120b import pd_singlehost

CONCURRENCY = 64
PREFILL_TOKENS = 8192


def pp_spec_role_configs(
    *,
    spec_args: dict[str, Any],
    spec_envs: dict[str, str],
    decode_args: dict[str, Any],
    decode_dp: int = 1,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """(GROUP, PREFILL_CONFIG, DECODE_CONFIG) serving spec_args on both engines.

    decode_dp > 1 runs the decode with DP attention over that many DP ranks.
    """
    group = deepcopy(pd_singlehost.GROUP)
    prefill = deepcopy(pd_singlehost.PREFILL_CONFIG)
    decode = deepcopy(pd_singlehost.DECODE_CONFIG)
    draft_model = spec_args["speculative_draft_model_path"]
    for config in (prefill, decode):
        config["server_args"].update(
            # Raiden's PP -> TP8 transfer gives each decode rank one kv head, and
            # FP8 packs need an even head count per rank.
            kv_cache_dtype="bfloat16",
            disable_radix_cache=True,
            mem_fraction_static=0.6,
            max_total_tokens=1187840,
            max_running_requests=CONCURRENCY,
            context_length=9280,
            chunked_prefill_size=PREFILL_TOKENS,
            max_prefill_tokens=PREFILL_TOKENS,
            torch_compile_max_bs=CONCURRENCY,
            tpu_compile_tokens=[PREFILL_TOKENS],
            tpu_compile_tracer="direct",
            trust_remote_code=True,
            watchdog_timeout=3600,
            dist_timeout=3600,
            disable_hybrid_swa_memory=True,
            **spec_args,
        )
        if config["server_args"]["model_path"].startswith("gs://"):
            config["server_args"]["load_format"] = "runai_streamer"
        if draft_model.startswith("gs://"):
            config["server_args"]["speculative_draft_load_format"] = "runai_streamer"
        config["runtime_envs"].update(
            SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER="1",
            SGLANG_RAIDEN_KV_PORT=os.environ.get("SGLANG_RAIDEN_KV_PORT", "40000"),
            SGLANG_WARMUP_TIMEOUT="3600",
            HEALTH_TIMEOUT=os.environ.get("WAIT_SECS", "5400"),
            **spec_envs,
        )

    # Eight single-device stages; PP rejects the overlap schedule.
    prefill["server_args"].update(
        tp_size=1, ep_size=1, pp_size=8, disable_overlap_schedule=True
    )

    # The perf test reads the decode's sglang:spec_accept_length gauge; the mini router has no /metrics.
    decode["server_args"]["enable_metrics"] = True
    decode["server_args"].update(decode_args)
    # A non-PP decode must name its layer pools to pull from a PP producer.
    decode["runtime_envs"]["SGLANG_RAIDEN_PIPELINE_TRANSFER"] = "1"
    # Spec uses CPU-tensor retraction; host-pool I/O is tested separately.
    decode["server_args"].pop("hicache_mem_layout")
    decode["server_args"].pop("hicache_io_backend")
    decode["server_args"]["disaggregation_decode_retraction_backup"] = "cpu_tensor"
    if decode_dp > 1:
        decode["server_args"].update(
            dp_size=decode_dp,
            enable_dp_attention=True,
            # The drafts share the target lm_head inside the attention-TP group.
            enable_dp_lm_head=True,
        )
        # Per-rank KV bytes per token change with DP; the profiler sizes the pool.
        decode["server_args"].pop("max_total_tokens")
        if decode["runtime_envs"].get("SGLANG_SPEC_TP_SYNC") == "rng":
            # Under DP attention both spec-accept sites sync, or neither.
            decode["runtime_envs"]["SGLANG_SPEC_TP_SYNC"] = "rng,spec-accept-greedy"
    return group, prefill, decode
