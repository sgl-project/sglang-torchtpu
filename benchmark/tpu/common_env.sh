#!/usr/bin/env bash
# Shared static environment for sglang torch_tpu serving. Sourced by
# launch_server.sh, so every recipe/benchmark that calls it gets the same env.
# Only *static* exports that are identical across launches live here; per-launch
# / per-model config (MODEL, TP_SIZE, CONTEXT_LEN, MEM_FRACTION_STATIC, ...) stays
# in launch_server.sh and the recipe launch scripts. Override any var by
# exporting it before launch.
#

# Interpreter with torch_tpu + sglang (editable) installed. Point at a venv if
# not on PATH, e.g. PYTHON=/path/to/.venv/bin/python.
export PYTHON=${PYTHON:-python3}

# Out-of-tree TPU platform plugin.
export SGLANG_PLATFORM=${SGLANG_PLATFORM:-tpu}

# HF cache: generic defaults (standard cache, online download). Hosts that keep
# weights on a local/GCS mount and run offline export HF_HOME + HF_HUB_OFFLINE=1
# + TRANSFORMERS_OFFLINE=1 themselves.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

# sglang / TPU serving knobs.
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=${SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION:-0}
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1}
# Per-phase scheduler/worker timing dumps every ~5 s (cheap).
export SGLANG_TPU_OVERLAP_PROFILE=${SGLANG_TPU_OVERLAP_PROFILE:-1}
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=${TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS:-false}

# Disable periodic diagnostic/telemetry harvesting in libtpu to prevent
# multihost core map std::out_of_range crashes during long serving runs.
export TPU_ENABLE_DIAGNOSTIC_COORDINATOR=${TPU_ENABLE_DIAGNOSTIC_COORDINATOR:-0}
export TPU_DISABLE_DIAGNOSTIC_COORDINATOR=${TPU_DISABLE_DIAGNOSTIC_COORDINATOR:-1}
export TPU_ENABLE_TELEMETRY_HARVESTING=${TPU_ENABLE_TELEMETRY_HARVESTING:-0}
export TPU_ENABLE_TELEMETRY=${TPU_ENABLE_TELEMETRY:-0}
export ENABLE_TPU_TELEMETRY=${ENABLE_TPU_TELEMETRY:-0}
export TPU_TELEMETRY_INTERVAL_SECONDS=${TPU_TELEMETRY_INTERVAL_SECONDS:-86400}
export TPU_TELEMETRY_INTERVAL_SEC=${TPU_TELEMETRY_INTERVAL_SEC:-86400}
export TPU_DIAGNOSTIC_HARVESTER_INTERVAL_SEC=${TPU_DIAGNOSTIC_HARVESTER_INTERVAL_SEC:-86400}

# Preload libtpu telemetry stub to prevent multihost std::out_of_range crashes in GetAllCoreStates()
_COMMON_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_STUB_SO="/tmp/libtpu_telemetry_stub.so"
if [[ ! -f "$_STUB_SO" && -f "$_COMMON_ENV_DIR/libtpu_telemetry_stub.c" ]]; then
    if command -v gcc >/dev/null 2>&1; then
        gcc -O2 -fPIC -shared "$_COMMON_ENV_DIR/libtpu_telemetry_stub.c" -o "$_STUB_SO" 2>/dev/null || true
    elif command -v clang >/dev/null 2>&1; then
        clang -O2 -fPIC -shared "$_COMMON_ENV_DIR/libtpu_telemetry_stub.c" -o "$_STUB_SO" 2>/dev/null || true
    fi
fi
if [[ -f "$_STUB_SO" ]]; then
    export LD_PRELOAD="${_STUB_SO}${LD_PRELOAD:+:$LD_PRELOAD}"
fi

# torch.compile logging: +recompiles surfaces recompilation churn cheaply. Avoid
# +dynamo / +output_code / +guards — they stream per-bytecode tracing logs and
# turn a ~minute compile into 10+ minutes plus a multi-GB log file.
export TORCH_LOGS=${TORCH_LOGS:-+recompiles}
# Serving-time XLA compiles; SITES attributes eager compiles to sglang lines.
# Off by default: every forward that misses the in-memory cache (persistent-cache
# loads included) syncs the device, which stalls spec decode most.
export SGLANG_TPU_CHECK_RECOMPILATION=${SGLANG_TPU_CHECK_RECOMPILATION:-0}
# Off by default: its dispatch mode costs ~40us per eager op (CPU ops included).
export SGLANG_TPU_CHECK_RECOMPILATION_SITES=${SGLANG_TPU_CHECK_RECOMPILATION_SITES:-0}

# XLA / libtpu tuning for TP/EP TPU serving. Latency-hiding scheduler + async
# collective fusion keep collectives off the matmul critical path; the post-SPMD
# collective-matmul modes and the SparseCore-offload=false flags are the
# DP-attention hardening set (avoid TP correctness issues / SparseCore halts).
# Overriding this string invalidates the compile cache.
# The Python launcher's copy is recipe/configs/base.py:LIBTPU_FLAGS.
export LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-"--xla_tpu_enable_async_collective_fusion=true \
--xla_tpu_overlap_compute_collective_tc=true \
--xla_tpu_enable_async_collective_fusion_multiple_steps=true \
--xla_tpu_enable_latency_hiding_scheduler=true \
--xla_latency_hiding_scheduler_rerun=10 \
--xla_tpu_all_gather_collective_matmul_mode=post_spmd_conservative \
--xla_tpu_reduce_scatter_collective_matmul_mode=post_spmd_conservative \
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false \
--xla_tpu_offload_gather_to_sparsecore=false \
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false"}
