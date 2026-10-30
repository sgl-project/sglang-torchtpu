"""Shared settings for TPU launch recipes."""

import os
from copy import deepcopy
from typing import Any

# Overlap compute with collectives; the sparse-core collective offloads stay off
# (they regressed these serving shapes). The joined string is part of the
# compile-cache key, so a derived recipe substitutes a flag in place rather than
# appending. common_env.sh carries the shell launcher's copy.
LIBTPU_FLAGS = (
    "--xla_tpu_enable_async_collective_fusion=true",
    "--xla_tpu_overlap_compute_collective_tc=true",
    "--xla_tpu_enable_async_collective_fusion_multiple_steps=true",
    "--xla_tpu_enable_latency_hiding_scheduler=true",
    "--xla_latency_hiding_scheduler_rerun=10",
    "--xla_tpu_all_gather_collective_matmul_mode=post_spmd_conservative",
    "--xla_tpu_reduce_scatter_collective_matmul_mode=post_spmd_conservative",
    "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false",
    "--xla_tpu_offload_gather_to_sparsecore=false",
    "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false",
)

CONFIG = {
    "server_args": {
        "device": "tpu",
        "dtype": "bfloat16",
        "host": "127.0.0.1",
        "port": 30000,
        "grammar_backend": "none",
    },
    # An inherited platform wins, as in common_env.sh: mode B runs with tpu_oot.
    "runtime_env": {"SGLANG_PLATFORM": os.environ.get("SGLANG_PLATFORM", "tpu")},
}

_FP8_BASE_CONFIG = deepcopy(CONFIG)
_FP8_BASE_CONFIG["server_args"].update(
    {
        "attention_backend": "pallas_tpu_rpa_v3",
        "page_size": 128,
        "max_running_requests": 64,
        "chunked_prefill_size": 4096,
        "max_prefill_tokens": 4096,
        "context_length": 9236,
        "torch_compile_max_bs": 64,
        "enable_torch_compile": True,
        "quantization": "fp8",
        "kv_cache_dtype": "fp8_e4m3",
        "disable_radix_cache": True,
    }
)
_FP8_BASE_CONFIG["runtime_env"].update(
    {
        # Online unless the caller set it, as common_env.sh does: an offline cluster
        # reads HF repo ids from a staged cache.
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "0"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "0"),
        "TORCH_LOGS": "+recompiles",
        "TPU_ENABLE_DIAGNOSTIC_COORDINATOR": "0",
        "TPU_DISABLE_DIAGNOSTIC_COORDINATOR": "1",
        "TPU_DIAGNOSTIC_HARVESTER_INTERVAL_SEC": "86400",
        "SGLANG_TPU_PRELOAD_TELEMETRY_STUB": "1",
        "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "0",
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        "SGLANG_TPU_OVERLAP_PROFILE": "1",
        "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": "false",
    }
)

_FP8_BASE_CONFIG["runtime_env"].update(
    {
        "TPU_ENABLE_TELEMETRY_HARVESTING": "0",
        "TPU_ENABLE_TELEMETRY": "0",
        "ENABLE_TPU_TELEMETRY": "0",
        "TPU_TELEMETRY_INTERVAL_SECONDS": "86400",
        "TPU_TELEMETRY_INTERVAL_SEC": "86400",
        "LIBTPU_INIT_ARGS": " ".join(LIBTPU_FLAGS),
    }
)


PD_SINGLEHOST_GROUP = {"hosts": 2, "prefill_engines": 1, "decode_engines": 1}


def pd_singlehost_role_configs(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One-host-per-engine raiden P/D roles serving config's model and flags.

    Both roles keep config's spec flags: the draft KV pool and the draft seed
    cross the handoff, so their layouts must match.

    Returns:
      (PREFILL_CONFIG, DECODE_CONFIG).
    """
    common = deepcopy(config)
    common["server_args"].update(
        host="0.0.0.0",
        disaggregation_transfer_backend="raiden",
    )
    common["runtime_env"].update(
        SGLANG_RAIDEN_KV_PORT=os.environ.get("SGLANG_RAIDEN_KV_PORT", "31000"),
        SGLANG_RAIDEN_RESHARD="0",
        LB_PORT=os.environ.get("LB_PORT", "8000"),
        HEALTH_TIMEOUT=os.environ.get("WAIT_SECS", "3600"),
    )
    prefill = deepcopy(common)
    prefill["server_args"].update(
        port=int(os.environ.get("PREFILL_PORT", "30000")),
        disaggregation_mode="prefill",
        disaggregation_bootstrap_port=int(os.environ.get("BOOTSTRAP_PORT", "8998")),
    )
    decode = deepcopy(common)
    decode["server_args"].update(
        port=int(os.environ.get("DECODE_PORT", "30000")),
        disaggregation_mode="decode",
        disaggregation_decode_retraction_backup="cpu_tensor",
    )
    return prefill, decode
