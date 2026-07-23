#!/usr/bin/env bash
# Model-agnostic launcher for an sglang torch_tpu server (--device tpu).
# Set MODEL (and TP_SIZE for multi-chip tensor parallelism); everything else
# has a sensible default. Pair with ./bench_serving.sh.
#
# Example — Qwen3-32B on all 8 chiplets (tp=8):
#   MODEL=Qwen/Qwen3-32B TP_SIZE=8 ./launch_server.sh
#
# Example — Qwen3 MoE FP8 EP throughput run (tp=8, ep=8):
#   MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 EP_SIZE=8 \
#   CONTEXT_LEN=2048 CHUNKED_PREFILL_SIZE=1024 MAX_PREFILL_TOKENS=1024 \
#   MAX_TOTAL_TOKENS=262144 MEM_FRACTION_STATIC=0.30 \
#   ./launch_server.sh
#
# Example — Qwen3-Coder-480B FP8 TP8/EP8, 1k/1k c=64:
#   MODEL=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 TP_SIZE=8 EP_SIZE=8 \
#   CONTEXT_LEN=2048 CHUNKED_PREFILL_SIZE=1024 MAX_PREFILL_TOKENS=1024 \
#   MAX_TOTAL_TOKENS=131072 MEM_FRACTION_STATIC=0.05 \
#   ./launch_server.sh
#
# Example — a small model on a single chip (tp=1; auto-pins chiplet 0):
#   MODEL=Qwen/Qwen3-0.6B TP_SIZE=1 PAGE_SIZE=32 ./launch_server.sh
#
# Example — multihost MoE across a 2x2x2 v7x slice (2 hosts → tp=16, ep=8).
# Preferred: Ray mode. Bring up a Ray cluster (head + workers), then run this
# ONLY on the head. The head Engine discovers the peer hostnames + slice
# topology from the live Ray cluster, so you supply only NNODES — no
# TPU_WORKER_HOSTNAMES, node rank, or dist addr. TP_SIZE auto-derives to
# NNODES*CHIPS_PER_HOST:
#   SGLANG_USE_RAY=1 NNODES=2 MODEL=Qwen/Qwen3-30B-A3B EP_SIZE=8 \
#   MEM_FRACTION_STATIC=0.30 ./launch_server.sh
#
# Alternative — no Ray: launch this SAME command on every host (one pod per
# host); node rank and the peer list are auto-detected from the GKE TPU env
# (TPU_WORKER_ID / TPU_WORKER_HOSTNAMES):
#   MODEL=Qwen/Qwen3-30B-A3B EP_SIZE=8 MEM_FRACTION_STATIC=0.30 ./launch_server.sh
# See the "multihost (multi-node) coordination" section below for manual control.
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
# Static environment (PYTHON, SGLANG_PLATFORM, HF cache, LIBTPU_INIT_ARGS,
# TORCH_LOGS, ...) is sourced from ./common_env.sh. See ./README.md for the full
# launch + bench recipe.
#
# Overridable env vars (with defaults shown):
#   MODEL=                         # REQUIRED — HF repo or local path
#   HOST=127.0.0.1
#   PORT=30000
#   TP_SIZE=8                      # tensor-parallel ranks (= chiplets)
#   EP_SIZE=                       # expert-parallel ranks; passed only when set.
#                                  # Default off (sglang ep_size=1: experts run
#                                  # tensor-parallel within the TP group). For a
#                                  # MoE model set EP to a divisor of TP_SIZE and
#                                  # the expert count, e.g. TP=16 -> EP=8 or 16.
#   CONC=64                        # bench concurrency this server is sized for
#   MAX_RUNNING=$CONC              # in-flight request cap (InferenceX parity:
#                                  # exactly the bench concurrency)
#   PAGE_SIZE=128
#   ISL=8192 OSL=1024              # bench shape, only used to derive CONTEXT_LEN
#   CONTEXT_LEN=$((ISL+OSL+20))    # tight context (InferenceX parity)
#   CHUNKED_PREFILL_SIZE=4096      # per-forward token cap
#   MAX_PREFILL_TOKENS=            # defaults to CHUNKED_PREFILL_SIZE
#   MAX_TOTAL_TOKENS=              # passed through if set
#   ATTENTION_BACKEND=pallas_tpu_rpa_v3
#   DISABLE_OVERLAP_SCHEDULE=     # 1 passes --disable-overlap-schedule
#   TPU_COMPILE_TOKENS=            # optional token bucket override
#   TPU_COMPILE_BS=                # optional bs bucket override
#   TORCH_COMPILE_MAX_BS=$MAX_RUNNING
#   NUM_CONTINUOUS_DECODE_STEPS=   # passed through if set
#   SGLANG_TPU_DECODE_PROCESS_INTERVAL=
#                                  # passed as env when set.
#   MEM_FRACTION_STATIC=           # passed through if set (sglang default
#                                  # 0.4; bump to 0.85 for high-concurrency
#                                  # to leave more HBM for KV cache)
#   QUANTIZATION=fp8               # --quantization. Set empty to disable
#                                  # (e.g. for bf16 checkpoints)
#   KV_CACHE_DTYPE=fp8_e4m3        # fp8 KV
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
#   --- multihost (multi-node serving) ---
#   SGLANG_USE_RAY=                # 1 → Ray mode: run only on the head; nnodes +
#                                  #   peer hostnames auto-discovered from the Ray
#                                  #   cluster. Supply only NNODES.
#   NNODES=                        # # hosts. Ray mode: user-supplied (may be
#                                  #   overridden by the discovered cluster).
#                                  #   Non-Ray: auto from TPU_WORKER_HOSTNAMES.
#   NODE_RANK=                     # non-Ray this host's rank; auto from TPU_WORKER_ID
#   DIST_INIT_ADDR=                # non-Ray torch.distributed rendezvous addr;
#                                  #   auto-built (Ray computes it at runtime)
#   DIST_PORT=10011                # rendezvous port (NOT the HTTP PORT)
#   CHIPS_PER_HOST=8               # tp-devices per host (v7x: 4 chips x 2 cores)
#   RANK0_HOST=                    # rank-0 hostname; needed for node_rank>0 manual mode
set -euo pipefail

# Shared static TPU env (PYTHON, SGLANG_PLATFORM, HF cache, LIBTPU_INIT_ARGS,
# TORCH_LOGS, ...) lives in common_env.sh next to this script — source it so
# every caller gets the same environment. Per-launch config stays below.
source "$(dirname "${BASH_SOURCE[0]}")/common_env.sh"

MODEL=${MODEL:-}
if [[ -z "$MODEL" ]]; then
    echo "ERROR: set MODEL, e.g. MODEL=Qwen/Qwen3-32B ./launch_server.sh" >&2
    exit 2
fi
# Slug for per-model cache / artifact dir names: lowercase, non-alnum → '_'.
MODEL_SLUG=$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | sed 's/__*/_/g; s/^_//; s/_$//')

# ===== multihost (multi-node) coordination =====
: "${DIST_PORT:=10011}"        # torch.distributed rendezvous port (NOT the HTTP PORT)
: "${CHIPS_PER_HOST:=8}"       # tp-devices per host (v7x: 4 chips x 2 cores)
USE_RAY=${SGLANG_USE_RAY:-0}
if [[ "$USE_RAY" == "1" ]]; then
    # Ray mode: only the HEAD runs this script. The head Engine discovers the
    # peer hostnames + slice topology straight from the live Ray cluster
    # (ray.nodes() + a SPREAD placement group that probes every node's
    # TPU_WORKER_ID / IP, then sorts by TPU_WORKER_ID for physical order).
    # So TPU_WORKER_HOSTNAMES / NODE_RANK / RANK0_HOST / DIST_INIT_ADDR are all
    # discovered at runtime and NOT needed here — the user supplies only NNODES.
    NNODES=${NNODES:-1}
    NODE_RANK=0
    [[ "$NNODES" -gt 1 ]] && ENABLE_MULTIHOST=1 || ENABLE_MULTIHOST=0
elif [[ -n "${TPU_WORKER_ID:-}" && -n "${TPU_WORKER_HOSTNAMES:-}" ]]; then
    ENABLE_MULTIHOST=1
    NODE_RANK=${NODE_RANK:-$TPU_WORKER_ID}
    NNODES=${NNODES:-$(awk -F, '{print NF}' <<< "$TPU_WORKER_HOSTNAMES")}
    RANK0_HOST=${RANK0_HOST:-$(cut -d, -f1 <<< "$TPU_WORKER_HOSTNAMES")}
elif [[ "${NNODES:-1}" -gt 1 ]]; then
    ENABLE_MULTIHOST=1             # manual: caller sets NNODES / NODE_RANK / RANK0_HOST
    NODE_RANK=${NODE_RANK:-0}
else
    ENABLE_MULTIHOST=0; NNODES=1; NODE_RANK=0
fi
if [[ "$ENABLE_MULTIHOST" == "1" ]]; then
    # tp counts DEVICES across the whole slice: NNODES * CHIPS_PER_HOST. In Ray
    # mode the user sets only NNODES; this gives a correct slice-wide default tp
    # (an explicit TP_SIZE=... still overrides below).
    DEFAULT_TP=$((NNODES * CHIPS_PER_HOST))
    if [[ "$USE_RAY" != "1" ]]; then
        # Non-Ray multihost: build the torch.distributed rendezvous addr here
        # (Ray computes it from the discovered rank-0 node IP instead).
        if [[ -z "${DIST_INIT_ADDR:-}" ]]; then
            if [[ "$NODE_RANK" == "0" ]]; then
                DIST_INIT_ADDR="0.0.0.0"
            else
                DIST_INIT_ADDR="${RANK0_HOST:?set RANK0_HOST or DIST_INIT_ADDR for node_rank>0}"
            fi
        fi
        [[ "$DIST_INIT_ADDR" == *:* ]] || DIST_INIT_ADDR="$DIST_INIT_ADDR:$DIST_PORT"
    fi
else
    DEFAULT_TP=8
fi

TP_SIZE=${TP_SIZE:-$DEFAULT_TP}
EP_SIZE=${EP_SIZE:-}

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=${RUN_DIR:-./profile/${MODEL_SLUG}-tpu/$TS}

ENABLE_HLO_DUMP=${ENABLE_HLO_DUMP:-0}
ENABLE_TORCH_COMPILE_DEBUG=${ENABLE_TORCH_COMPILE_DEBUG:-0}
# 1 (default) → pass --enable-torch-compile. Set 0 to run eager (debugging:
# isolate compile bugs and allow .item() probes without graph breaks).
ENABLE_TORCH_COMPILE=${ENABLE_TORCH_COMPILE:-1}
ENABLE_PROFILER_ENDPOINT=${ENABLE_PROFILER_ENDPOINT:-1}
# Optional JSON merged into the HF config via --json-model-override-args, e.g.
# '{"vision_config":null,"audio_config":null}' to serve a multimodal checkpoint
# text-only. Passed as one quoted argv element (see the args+= block below).
JSON_MODEL_OVERRIDE=${JSON_MODEL_OVERRIDE:-}

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

# SGLANG_PLATFORM / SGLANG_TPU_OVERLAP_PROFILE / MATERIALIZE_COLLECTIVE_TENSORS
# are exported by common_env.sh (sourced above).
if [[ -n "${SGLANG_TPU_DECODE_PROCESS_INTERVAL:-}" ]]; then
    export SGLANG_TPU_DECODE_PROCESS_INTERVAL
fi

# ===== torch.compile debug =====
# TORCH_LOGS is exported by common_env.sh.
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

# LIBTPU_INIT_ARGS (XLA/libtpu TP/EP serving tuning + DP-attention hardening) is
# exported by common_env.sh. Override that var to retune (invalidates the cache).

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
# InferenceX parity: size the server to the bench leg. max-running-requests is
# exactly the bench concurrency and context-length is a tight ISL+OSL+20.
CONC=${CONC:-64}
MAX_RUNNING=${MAX_RUNNING:-$CONC}
PAGE_SIZE=${PAGE_SIZE:-128}
ISL=${ISL:-8192}
OSL=${OSL:-1024}
CONTEXT_LEN=${CONTEXT_LEN:-$((ISL + OSL + 20))}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-pallas_tpu_rpa_v3}
# Pin the runtime prefill cap so TPU's derived token bucket grid is stable.
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-4096}
MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-$CHUNKED_PREFILL_SIZE}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-}
# Maximum batch size cap for torch.compile and bucketing grids.
TORCH_COMPILE_MAX_BS=${TORCH_COMPILE_MAX_BS:-$MAX_RUNNING}
# Optional shape bucket overrides for the TPU precompile + pad shim. Leave
# unset to use the backend's derived powers-of-two buckets.
TPU_COMPILE_TOKENS=${TPU_COMPILE_TOKENS:-}
TPU_COMPILE_BS=${TPU_COMPILE_BS:-}
NUM_CONTINUOUS_DECODE_STEPS=${NUM_CONTINUOUS_DECODE_STEPS:-}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-}
# InferenceX parity: FP8 weights + FP8 KV cache by default. Set either to the
# empty string to fall back to the sglang default (e.g. bf16 checkpoints).
# Use ${VAR-default} (no colon) so an explicit empty value disables it. Having
# colon would treat empty variable as unset and incorrectly re-apply default
# value specified here.
QUANTIZATION=${QUANTIZATION-fp8}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE-fp8_e4m3}
# 1 (default, InferenceX parity) → pass --disable-radix-cache. Benchmark
# prompts share synthetic prefixes, so radix cache would short-circuit
# prefills and inflate results (it also hides larger-bucket compiles when
# validating warmup scripts).
DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-1}

echo "==== launch config ===="
echo "  model              = $MODEL"
echo "  model_slug         = $MODEL_SLUG"
echo "  tp_size            = $TP_SIZE"
if [[ "$ENABLE_MULTIHOST" == "1" ]]; then
  if [[ "$USE_RAY" == "1" ]]; then
    echo "  multihost          = ray-discovered (nnodes hint=$NNODES, chips/host=$CHIPS_PER_HOST)"
  else
    echo "  multihost          = nnodes=$NNODES node_rank=$NODE_RANK dist=$DIST_INIT_ADDR (chips/host=$CHIPS_PER_HOST)"
  fi
fi
echo "  ep_size            = ${EP_SIZE:-<unset>}"
echo "  host:port          = $HOST:$PORT"
echo "  max_running        = $MAX_RUNNING  (CONC=$CONC)"
echo "  page_size          = $PAGE_SIZE"
echo "  context_len        = $CONTEXT_LEN  (ISL=$ISL OSL=$OSL)"
echo "  attention_backend  = $ATTENTION_BACKEND"
echo "  chunked_prefill    = $CHUNKED_PREFILL_SIZE"
echo "  max_prefill_tokens = ${MAX_PREFILL_TOKENS:-<unset>}"
echo "  max_total_tokens   = ${MAX_TOTAL_TOKENS:-<unset>}"
echo "  tpu_compile_tokens = ${TPU_COMPILE_TOKENS:-<derived>}"
echo "  tpu_compile_bs     = ${TPU_COMPILE_BS:-<derived>}"
echo "  num_cont_decode    = ${NUM_CONTINUOUS_DECODE_STEPS:-<unset>}"
echo "  mem_fraction_static= ${MEM_FRACTION_STATIC:-<sglang default>}"
echo "  quantization       = ${QUANTIZATION:-<sglang default>}"
echo "  kv_cache_dtype     = ${KV_CACHE_DTYPE:-<sglang default (auto)>}"
echo "  torch_compile_max_bs = $TORCH_COMPILE_MAX_BS"
echo "  disable_radix_cache  = $DISABLE_RADIX_CACHE"
echo "  disable_overlap_schedule = ${DISABLE_OVERLAP_SCHEDULE:-<platform default>}"
echo "  tpu_decode_process_interval = ${SGLANG_TPU_DECODE_PROCESS_INTERVAL:-<unset>}"
echo "  compile_cache        = $ENABLE_COMPILE_CACHE  (tier2=${TORCH_TPU_TIER2_NAME}, tier3=${TORCH_TPU_TIER3_ROOT})"
echo "  materialize_collective_tensors = $TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS"
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
    --attention-backend "$ATTENTION_BACKEND"
    --tp-size "$TP_SIZE"
    --grammar-backend none
    --host "$HOST"
    --port "$PORT"
    --max-running-requests "$MAX_RUNNING"
    --page-size "$PAGE_SIZE"
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
)
[[ -n "$TPU_COMPILE_TOKENS" ]] && args+=(--tpu-compile-tokens $TPU_COMPILE_TOKENS)
[[ -n "$TPU_COMPILE_BS" ]] && args+=(--tpu-compile-bs $TPU_COMPILE_BS)
[[ -n "$NUM_CONTINUOUS_DECODE_STEPS" ]] && args+=(--num-continuous-decode-steps "$NUM_CONTINUOUS_DECODE_STEPS")
[[ -n "$EP_SIZE" ]] && args+=(--ep-size "$EP_SIZE")
[[ -n "$MEM_FRACTION_STATIC" ]] && args+=(--mem-fraction-static "$MEM_FRACTION_STATIC")
[[ -n "$QUANTIZATION" ]] && args+=(--quantization "$QUANTIZATION")
[[ -n "$KV_CACHE_DTYPE" ]] && args+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
[[ -n "$TORCH_COMPILE_MAX_BS" ]] && args+=(--torch-compile-max-bs "$TORCH_COMPILE_MAX_BS")
[[ -n "$CONTEXT_LEN" ]] && args+=(--context-length "$CONTEXT_LEN")
[[ -n "$MAX_PREFILL_TOKENS" ]] && args+=(--max-prefill-tokens "$MAX_PREFILL_TOKENS")
[[ -n "$MAX_TOTAL_TOKENS" ]] && args+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
[[ "$DISABLE_RADIX_CACHE" == "1" ]] && args+=(--disable-radix-cache)
[[ "$ENABLE_TORCH_COMPILE" == "1" ]] && args+=(--enable-torch-compile)
[[ -n "${JSON_MODEL_OVERRIDE:-}" ]] && args+=(--json-model-override-args "$JSON_MODEL_OVERRIDE")
[[ "${DISABLE_OVERLAP_SCHEDULE:-0}" == "1" ]] && args+=(--disable-overlap-schedule)
# Rendezvous flags. In Ray mode NNODES is the source of truth (default 1; an
# explicit NNODES wins) and is always passed with --use-ray, independent of
# ENABLE_MULTIHOST -- Ray discovers node ranks and the rendezvous addr from the
# live cluster, so no --node-rank / --dist-init-addr. (NNODES=0 opts into
# head-side auto-detection of the node count.) Off Ray, the multinode flags are
# only needed when fanned across hosts.
if [[ "$USE_RAY" == "1" ]]; then
    args+=(--nnodes "$NNODES" --use-ray)
elif [[ "$ENABLE_MULTIHOST" == "1" ]]; then
    args+=(--nnodes "$NNODES" --node-rank "$NODE_RANK" --dist-init-addr "$DIST_INIT_ADDR")
fi
# KV offloading to host RAM (hierarchical cache); orthogonal to multihost.
[[ "${KV_OFFLOAD:-0}" == "1" ]] && args+=(--enable-hierarchical-cache --hicache-ratio "${HICACHE_RATIO:-2.0}")

# Escape hatch for any launch_server flag not surfaced above, e.g.
#   EXTRA_ARGS="--weight-loader-disable-mmap" MODEL=... ./launch_server.sh
# (--weight-loader-disable-mmap reads each safetensors file with a plain
# f.read() instead of mmap; needed when weights live on a GCS-FUSE mount,
# where mmap page faults can SIGBUS the loader). Word-split intentionally.
EXTRA_ARGS=${EXTRA_ARGS:-}
[[ -n "$EXTRA_ARGS" ]] && args+=($EXTRA_ARGS)

# PYTHON is exported by common_env.sh (default python3).
set -x
exec "$PYTHON" "${args[@]}" 2>&1
set +x
