"""Muse Spark AgentX Raiden P/D: a PP8 prefill host feeding a TP8 decode host.

Both engines start from agentic.py's fp8 SEQ_ALONG_LANE build and run the
experimental spec worker, the only one serving DFlash on a PP prefill. Both keep
a radix + HiCache prefix cache, so a multi-turn hit on the decode skips resending
its KV; the decode runs the megakernel.

    bash benchmark/tpu/recipe/muse_spark_agentx_pd_v7_4chip.sh
"""

from __future__ import annotations

from benchmark.tpu.recipe.configs.muse_spark.agentic import build_config

GROUP = {"hosts": 2, "prefill_engines": 1, "decode_engines": 1}

# The megakernel caps a decode step at 48 rows: 6 requests x 8 draft tokens.
MAX_RUNNING_REQUESTS = 6


def _role_config() -> dict:
    config = build_config(MAX_RUNNING_REQUESTS, mk_max_rows=48)
    config["server_args"].update(
        disaggregation_transfer_backend="raiden",
        watchdog_timeout=3600,
        dist_timeout=3600,
    )
    config["runtime_env"].update(
        SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER="1",
        # The experimental worker only simulates acceptance with fixed draft tokens.
        SGLANG_SIMULATE_ACC_TOKEN_MODE="fixed",
        SGLANG_WARMUP_TIMEOUT="3600",
    )
    return config


PREFILL_CONFIG = _role_config()
PREFILL_CONFIG["server_args"].update(
    disaggregation_mode="prefill",
    # Raiden's pipeline producer needs attention TP=1; PP rejects the overlap schedule.
    tp_size=1,
    ep_size=1,
    pp_size=8,
    disable_overlap_schedule=True,
)
# The megakernel runs the whole decoder in one kernel, so it cannot be staged.
PREFILL_CONFIG["runtime_env"]["SGLANG_TPU_MUSE_MEGAKERNEL"] = "0"

DECODE_CONFIG = _role_config()
DECODE_CONFIG["server_args"].update(
    disaggregation_mode="decode",
    # The decode HiCache (pd_decode_swa_hicache) rejects host_pool retraction.
    disaggregation_decode_retraction_backup="cpu_tensor",
    # The perf harness reads sglang:spec_accept_length from the decode.
    enable_metrics=True,
)
# A non-PP decode names its layer pools to pull from the PP producer.
DECODE_CONFIG["runtime_env"]["SGLANG_RAIDEN_PIPELINE_TRANSFER"] = "1"
