#!/usr/bin/env bash
# Model-agnostic launcher for an sglang torch_tpu server (--device tpu).
# Set MODEL (and TP_SIZE for multi-chip tensor parallelism); everything else
# has a sensible default. Pair with ./bench_serving.sh.
#
# Example — Qwen3-32B on all 8 chiplets (tp=8):
#   MODEL=Qwen/Qwen3-32B TP_SIZE=8 ./launch_server.sh
#
# Example — a small model on a single chip (tp=1; auto-pins chiplet 0):
#   MODEL=Qwen/Qwen3-0.6B TP_SIZE=1 PAGE_SIZE=32 ./launch_server.sh
#
# Chiplet visibility is derived from TP_SIZE: tp=1 pins to chiplet 0 (torch_tpu
# needs exactly one addressable PjRt device at process-group init), tp>1 leaves
# all chiplets visible so the TPU backend's _prepare_tpu_env(tp) lays out one
# PjRt rank per chiplet.
#
# Heavy debug dumps (HLO, torch.compile FX) are OFF by default — they add
# noticeable per-compile latency and ~100s of MB of disk. Re-enable when you
# actually need to inspect graphs:
#   ENABLE_HLO_DUMP=1 ENABLE_TORCH_COMPILE_DEBUG=1 MODEL=... ./launch_server.sh
#
# See ./README.md for the shared env contract (PYTHON, JAX_BAZEL, HF cache,
# compile cache) and the full launch + bench recipe.
#
# Overridable env vars (with defaults shown):
#   MODEL=                         # REQUIRED — HF repo or local path
#   HOST=127.0.0.1
#   PORT=30000
#   TP_SIZE=8                      # tensor-parallel ranks (= chiplets)
#   MAX_RUNNING=64                 # in-flight request cap (>= bench concurrency)
#   PAGE_SIZE=64
#   MEM_FRACTION_STATIC=           # passed through if set (sglang default
#                                  # 0.4; bump to 0.85 for high-concurrency
#                                  # to leave more HBM for KV cache)
#   KV_CACHE_DTYPE=                # passed through if set; "fp8" halves
#                                  # KV traffic and roughly doubles per-step
#                                  # decode speed for memory-bound regimes.
#                                  # Leave empty for sglang default (auto).
#   ENABLE_HLO_DUMP=0              # 1 → dump HLO modules to $HLO_DUMP_DIR
#   ENABLE_TORCH_COMPILE_DEBUG=0   # 1 → dump FX/lowered to $TORCH_COMPILE_DEBUG_DIR
#   ENABLE_PROFILER_ENDPOINT=1     # 1 → set SGLANG_TORCH_PROFILER_DIR so
#                                  # /start_profile produces traces
#   RUN_DIR=./profile/<model-slug>-tpu/<timestamp>
#   PROFILER_DIR=$RUN_DIR
#   HLO_DUMP_DIR=./hlo_dump/<model-slug>-tpu/<timestamp>
#   TORCH_COMPILE_DEBUG_DIR=$RUN_DIR/torch_compile_debug
#   TORCH_TPU_TIER2_NAME=torchtpu_<model-slug>   # per-model compile cache name
#   PYTHON=python3                 # interpreter with torch_tpu + sglang installed
#   JAX_BAZEL=                     # opt-in; only for a Bazel-built torch_tpu (see below)
set -euo pipefail

MODEL=${MODEL:-}
if [[ -z "$MODEL" ]]; then
    echo "ERROR: set MODEL, e.g. MODEL=Qwen/Qwen3-32B ./launch_server.sh" >&2
    exit 2
fi
# Slug for per-model cache / artifact dir names: lowercase, non-alnum → '_'.
MODEL_SLUG=$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | sed 's/__*/_/g; s/^_//; s/_$//')

TP_SIZE=${TP_SIZE:-8}

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=${RUN_DIR:-./profile/${MODEL_SLUG}-tpu/$TS}

ENABLE_HLO_DUMP=${ENABLE_HLO_DUMP:-0}
ENABLE_TORCH_COMPILE_DEBUG=${ENABLE_TORCH_COMPILE_DEBUG:-0}
ENABLE_PROFILER_ENDPOINT=${ENABLE_PROFILER_ENDPOINT:-1}

PROFILER_DIR=${PROFILER_DIR:-$RUN_DIR}
HLO_DUMP_DIR=${HLO_DUMP_DIR:-./hlo_dump/${MODEL_SLUG}-tpu/$TS}
TORCH_COMPILE_DEBUG_DIR=${TORCH_COMPILE_DEBUG_DIR:-$RUN_DIR/torch_compile_debug}

# Only create dirs for the artifacts we'll actually produce.
[[ "$ENABLE_PROFILER_ENDPOINT" == "1" ]] && mkdir -p "$PROFILER_DIR"
[[ "$ENABLE_HLO_DUMP" == "1" ]] && mkdir -p "$HLO_DUMP_DIR"
[[ "$ENABLE_TORCH_COMPILE_DEBUG" == "1" ]] && mkdir -p "$TORCH_COMPILE_DEBUG_DIR"

# A torch_tpu built from source with Bazel keeps jax/jaxlib/ml_dtypes/
# opt_einsum/scipy in its Bazel external repo rather than the venv's
# site-packages, so `python -m sglang.launch_server` can't import them. If you
# built that way, point JAX_BAZEL at <bazel-output-user-root>/<hash>/external
# and this prepends the five site-packages dirs to PYTHONPATH. Pip-installed
# torch_tpu does NOT need this — leave JAX_BAZEL unset.
JAX_BAZEL=${JAX_BAZEL:-}
if [[ -n "$JAX_BAZEL" ]]; then
    _jax_pp=""
    for _pkg in jax jaxlib ml_dtypes opt_einsum scipy; do
        _jax_pp="${_jax_pp:+$_jax_pp:}$JAX_BAZEL/rules_python++pip+torch_tpu_pypi_312_${_pkg}/site-packages"
    done
    export PYTHONPATH="${_jax_pp}${PYTHONPATH:+:$PYTHONPATH}"
fi

# Chiplet visibility, derived from TP_SIZE.
if [[ "$TP_SIZE" == "1" ]]; then
    # tp=1: torch_tpu requires exactly one addressable PjRt device at
    # process-group init (process_group_tpu.cc: "expected exactly one TPU
    # device per PyTorch process."). On a multichip host the PjRt client would
    # otherwise see all chiplets — pin to chiplet 0 with single-chip bounds.
    export TPU_VISIBLE_CHIPS=${TPU_VISIBLE_CHIPS:-0}
    export TPU_CHIPS_PER_HOST_BOUNDS=${TPU_CHIPS_PER_HOST_BOUNDS:-1,1,1}
    export TPU_HOST_BOUNDS=${TPU_HOST_BOUNDS:-1,1,1}
else
    # tp>1: leave all chiplets visible. hardware_backend/tpu/patches.py::
    # _prepare_tpu_env(world_size=TP_SIZE) sets TORCH_TPU_TOPOLOGY and
    # TORCH_TPU_SLICEBUILDER_ADDRESSES per-worker, so do NOT pin
    # TPU_VISIBLE_CHIPS / TPU_CHIPS_PER_HOST_BOUNDS — that would mask chiplets
    # and break the slicebuilder handshake.
    unset TPU_VISIBLE_CHIPS TPU_CHIPS_PER_HOST_BOUNDS TPU_HOST_BOUNDS 2>/dev/null || true
fi

export SGLANG_PLATFORM=tpu
# Per-phase scheduler/worker timing dumps every ~5 s (cheap).
export SGLANG_TPU_OVERLAP_PROFILE=${SGLANG_TPU_OVERLAP_PROFILE:-1}

# Materialize collective tensors so the c10d_functional all_reduce in the
# vocab-parallel embedding is traceable under torch.compile(fullgraph=True).
# Without this, TP>1 + --enable-torch-compile crashes on the first prefill with
# torch._dynamo.exc.Unsupported (the collective resolves to a raw pybind11
# function Dynamo marks as skipped). Required for the compiled TP>1 path.
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=${TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS:-0}

# HF cache. Defaults here are generic (standard HF cache, online download).
# The team's TPU hosts keep weights on a GCS-FUSE mount and run fully offline;
# export HF_HOME / HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE to match — see
# ./README.md.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

# ===== torch.compile debug =====
# +recompiles surfaces recompilation churn cheaply. Avoid +dynamo /
# +output_code / +guards here — those stream per-bytecode tracing logs
# synchronously and turn a ~minute compile into 10+ minutes plus a
# multi-GB log file.
export TORCH_LOGS=${TORCH_LOGS:-+recompiles}
if [[ "$ENABLE_TORCH_COMPILE_DEBUG" == "1" ]]; then
    # FX + lowered graphs per compile (~100s of MB at 32B/tp=8).
    export TORCH_COMPILE_DEBUG=1
    export TORCH_COMPILE_DEBUG_DIR
fi

# ===== torch_tpu compilation cache (Tier-2 in /dev/shm + Tier-3 on disk) =====
# Caches HLO→TPU binaries across server restarts. On cache hit, per-shape
# compile drops from ~30s to ~1s. Invalidated by changes to LIBTPU_INIT_ARGS,
# HLO bytes, libtpu version, or topology — so re-warm whenever flags move.
# The Tier-2 name defaults per-model so distinct models don't collide; Tier-3
# expects a full path (point it at a persistent local disk).
ENABLE_COMPILE_CACHE=${ENABLE_COMPILE_CACHE:-1}
TORCH_TPU_TIER2_NAME=${TORCH_TPU_TIER2_NAME:-torchtpu_${MODEL_SLUG}}
TORCH_TPU_TIER3_ROOT=${TORCH_TPU_TIER3_ROOT:-$HOME/.cache/torch_tpu_compile_cache}
if [[ "$ENABLE_COMPILE_CACHE" == "1" ]]; then
    mkdir -p "$TORCH_TPU_TIER3_ROOT"
    mkdir -p "/dev/shm/torch_tpu_cache/$TORCH_TPU_TIER2_NAME" 2>/dev/null || true
    export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE="$TORCH_TPU_TIER2_NAME"
    export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT="$TORCH_TPU_TIER3_ROOT"
    export TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK=1
fi

# ===== XLA / libtpu init =====
# XLA/libtpu tuning flags (latency-hiding scheduler + async collective fusion
# keep collectives from serializing behind matmuls on TPU v7). Override the
# whole string to retune. Changing it invalidates the compile cache (see above).
export LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-"--xla_tpu_scoped_vmem_limit_kib=32768 \
--xla_tpu_enable_async_collective_fusion=true \
--xla_tpu_overlap_compute_collective_tc=true \
--xla_tpu_enable_async_collective_fusion_multiple_steps=true \
--xla_tpu_enable_latency_hiding_scheduler=true"}

# ===== XLA HLO dump (opt-in) =====
# Match on `tt_jit.*` — torch_tpu compiles are named `tt_jit_<op>`; a
# `.*forward.*` filter (the obvious guess) catches none of them. The bare
# `tt_jit` module is the main model forward graph.
if [[ "$ENABLE_HLO_DUMP" == "1" ]]; then
    export XLA_FLAGS=${XLA_FLAGS:-"--xla_dump_to=$HLO_DUMP_DIR \
--xla_dump_hlo_as_text \
--xla_dump_hlo_as_proto \
--xla_dump_hlo_module_re=.*tt_jit.*"}
fi

# ===== torch profiler endpoint =====
if [[ "$ENABLE_PROFILER_ENDPOINT" == "1" ]]; then
    export SGLANG_TORCH_PROFILER_DIR="$PROFILER_DIR"
fi

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-30000}
MAX_RUNNING=${MAX_RUNNING:-64}
PAGE_SIZE=${PAGE_SIZE:-64}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-}
# 1 → pass --disable-radix-cache to sglang. Useful when validating warmup
# scripts: with radix on, repeated synthetic prompts share a long prefix
# and sglang short-circuits the prefill, hiding the larger-bucket compiles.
DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-0}
# sglang's torch.compile only pre-warms decode batch sizes up to this
# value at server startup. If the bench's max-concurrency exceeds it,
# the first batch-N decode forward at runtime triggers a fresh compile
# (one-off latency bump on the first wave's TTFT). Default (empty) keeps
# sglang's own default of 32; bump to the largest concurrency you'll bench
# against so all measurement buckets are pre-compiled.
TORCH_COMPILE_MAX_BS=${TORCH_COMPILE_MAX_BS:-}

echo "==== launch config ===="
echo "  model              = $MODEL"
echo "  model_slug         = $MODEL_SLUG"
echo "  tp_size            = $TP_SIZE"
echo "  host:port          = $HOST:$PORT"
echo "  max_running        = $MAX_RUNNING"
echo "  page_size          = $PAGE_SIZE"
echo "  mem_fraction_static= ${MEM_FRACTION_STATIC:-<sglang default>}"
echo "  kv_cache_dtype     = ${KV_CACHE_DTYPE:-<sglang default (auto)>}"
echo "  torch_compile_max_bs = ${TORCH_COMPILE_MAX_BS:-<sglang default (32)>}"
echo "  disable_radix_cache  = $DISABLE_RADIX_CACHE"
echo "  compile_cache        = $ENABLE_COMPILE_CACHE  (tier2=${TORCH_TPU_TIER2_NAME}, tier3=${TORCH_TPU_TIER3_ROOT})"
echo "  ENABLE_HLO_DUMP        = $ENABLE_HLO_DUMP  ($HLO_DUMP_DIR)"
echo "  ENABLE_TORCH_COMPILE_DEBUG = $ENABLE_TORCH_COMPILE_DEBUG  ($TORCH_COMPILE_DEBUG_DIR)"
echo "  ENABLE_PROFILER_ENDPOINT   = $ENABLE_PROFILER_ENDPOINT  ($PROFILER_DIR)"
echo "  TORCH_LOGS         = $TORCH_LOGS"
echo "======================="

# Build the launch_server arg list, only adding optional flags when set.
args=(
    -u -m sglang.launch_server
    --model-path "$MODEL"
    --dtype bfloat16
    --device tpu
    --tp "$TP_SIZE"
    --grammar-backend none
    --enable-torch-compile
    --host "$HOST"
    --port "$PORT"
    --max-running-requests "$MAX_RUNNING"
    --page-size "$PAGE_SIZE"
)
[[ -n "$MEM_FRACTION_STATIC" ]] && args+=(--mem-fraction-static "$MEM_FRACTION_STATIC")
[[ -n "$KV_CACHE_DTYPE" ]] && args+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
[[ -n "$TORCH_COMPILE_MAX_BS" ]] && args+=(--torch-compile-max-bs "$TORCH_COMPILE_MAX_BS")
[[ "$DISABLE_RADIX_CACHE" == "1" ]] && args+=(--disable-radix-cache)

# Escape hatch for any launch_server flag not surfaced above, e.g.
#   EXTRA_ARGS="--weight-loader-disable-mmap" MODEL=... ./launch_server.sh
# (--weight-loader-disable-mmap reads each safetensors file with a plain
# f.read() instead of mmap; needed when weights live on a GCS-FUSE mount,
# where mmap page faults can SIGBUS the loader). Word-split intentionally.
EXTRA_ARGS=${EXTRA_ARGS:-}
[[ -n "$EXTRA_ARGS" ]] && args+=($EXTRA_ARGS)

PYTHON=${PYTHON:-python3}
exec "$PYTHON" "${args[@]}" 2>&1
