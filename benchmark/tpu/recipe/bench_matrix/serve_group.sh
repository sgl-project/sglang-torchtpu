#!/usr/bin/env bash
# Launch-group entry point for the TPU benchmark matrix. The CI JobSet runs
# this on EVERY host of the slice (via a thin per-group wrapper that sets
# BENCH_MODEL/BENCH_RECIPE); what a host does depends on the recipe and its
# TPU_WORKER_ID:
#
#   1host_*  rank 0 serves a single-host tp=8 engine on :30000
#            (FORCE_SINGLE_HOST); every other rank exits 0 and leaves its
#            host idle (the registered suite's smallest slice is 2 hosts, so
#            a 1-host recipe deliberately strands the second).
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/config.sh"
TPU_BENCH="$TPU_BENCH_DIR"

WORKER_ID=${TPU_WORKER_ID:-0}

# Keep the torch_tpu tier-3 compile cache off the pod's ephemeral overlay:
# its $HOME default has filled the 40-80Gi overlay and had pods EVICTED,
# gang-restarting the whole JobSet.
export TORCH_TPU_TIER3_ROOT=${TORCH_TPU_TIER3_ROOT:-/dev/shm/torch_tpu_tier3}

# Env every engine shares, translated to launch_server.sh's interface.
export MODEL QUANTIZATION KV_CACHE_DTYPE ATTENTION_BACKEND PAGE_SIZE
export CONTEXT_LEN MEM_FRACTION_STATIC MAX_TOTAL_TOKENS
export CHUNKED_PREFILL_SIZE
export MAX_PREFILL_TOKENS=$CHUNKED_PREFILL_SIZE
export MAX_RUNNING TORCH_COMPILE_MAX_BS
export ENABLE_METRICS ENABLE_CACHE_REPORT
export ENABLE_TORCH_COMPILE=${ENABLE_TORCH_COMPILE:-1}
export DISABLE_RADIX_CACHE=$(( PREFIX_CACHE == 1 ? 0 : 1 ))
export PYTHON

BASE_EXTRA_ARGS="$MODEL_EXTRA_ARGS ${EXTRA_ARGS:-}"
[[ "$SPEC_DECODE" == "1" ]] && BASE_EXTRA_ARGS="$BASE_EXTRA_ARGS $SPEC_ARGS --sampling-backend pytorch"

log_config

# Strip the slice-wide multihost env before launching any SINGLE-host engine:
# libtpu on GKE is env-driven (TPU_SKIP_MDS_QUERY=true), and the injected
# TPU_WORKER_HOSTNAMES / TPU_PROCESS_ADDRESSES / TPU_TOPOLOGY describe the
# WHOLE slice -- _prepare_tpu_env would build a slice-wide slicebuilder that
# never forms (idle peers), and FORCE_SINGLE_HOST only stops launch_server's
# own nnodes logic, not that env detection. Local hardware discovery then
# resolves the single-host topology.
strip_slice_env() {
    unset TPU_WORKER_ID TPU_WORKER_HOSTNAMES TPU_PROCESS_ADDRESSES \
          TPU_TOPOLOGY TPU_TOPOLOGY_WRAP TPU_TOPOLOGY_ALT TPU_HOST_BOUNDS \
          TPU_ACCELERATOR_TYPE 2>/dev/null || true
}

if [[ "$WORKER_ID" != "0" ]]; then
    echo "[serve_group] rank $WORKER_ID idle for 1-host recipe; exiting 0"
    exit 0
fi
strip_slice_env
export FORCE_SINGLE_HOST=1
export TP_SIZE EP_SIZE
export PORT HOST=${HOST:-0.0.0.0}
export KV_OFFLOAD HICACHE_RATIO HICACHE_IO_BACKEND HICACHE_MEM_LAYOUT
export EXTRA_ARGS="$BASE_EXTRA_ARGS"
exec "$TPU_BENCH/launch_server.sh"
