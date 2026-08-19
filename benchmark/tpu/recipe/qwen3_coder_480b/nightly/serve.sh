#!/usr/bin/env bash
# Launch one engine for the Qwen3-Coder-480B FP8 TPU agentic benchmark.
# Thin wrapper over benchmark/tpu/launch_server.sh: it only translates the
# contract in config.sh into that script's env interface, so there is exactly
# one place where server flags are decided.
#
#   ROLE=colocated ./serve.sh      # tp=16 dp=4 (attn_tp=4), one engine
#   ROLE=prefill   ./serve.sh      # tp=8  dp=2 (attn_tp=4), P/D prefill
#   ROLE=decode    ./serve.sh      # tp=8  dp=2 (attn_tp=4), P/D decode
#
# The colocated role spans 2 TPU hosts (tp=16 = 2 x 8 chiplets). Ray mode is the
# easy path -- bring up a Ray cluster over both hosts, then run this ONLY on the
# head and launch_server.sh discovers the peers:
#   SGLANG_USE_RAY=1 NNODES=2 ROLE=colocated ./serve.sh
# Without Ray, run the identical command on every host; node rank and the peer
# list come from the GKE TPU env (TPU_WORKER_ID / TPU_WORKER_HOSTNAMES).
#
# Each P/D role is a single 8-chiplet host, so prefill and decode run on
# different hosts; pair them with a router (run_nightly.sh starts one).
#
# Env: everything in config.sh, plus
#   ROLE=colocated|prefill|decode
#   PORT / PD_PREFILL_PORT / PD_DECODE_PORT
#   LOG_FILE=          # tee the server's stdout+stderr here as well
#   RUN_DIR=           # artifact dir (profiles, launch_meta.json)
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/config.sh"

ROLE=${ROLE:-colocated}

case "$ROLE" in
    colocated)
        export TP_SIZE=$COLOCATED_TP
        export DP_SIZE=$COLOCATED_DP
        export EP_SIZE=$EP_SIZE_COLOCATED
        export PORT=$PORT
        ROLE_ARGS=""
        ;;
    prefill)
        export TP_SIZE=$PD_TP
        export DP_SIZE=$PD_DP
        export EP_SIZE=$EP_SIZE_PD
        export PORT=$PD_PREFILL_PORT
        # This engine occupies one host; the peer host runs the *other* engine,
        # not another TP rank of this one. Without this, launch_server.sh's GKE
        # auto-detection sees TPU_WORKER_HOSTNAMES and builds a 2-host
        # rendezvous that would never form.
        export FORCE_SINGLE_HOST=${FORCE_SINGLE_HOST:-1}
        # The TPU transfer backend moves KV over the torch_tpu DMA path; the
        # bootstrap port is how the decode engine finds this one.
        ROLE_ARGS="--disaggregation-mode prefill \
--disaggregation-transfer-backend tpu \
--disaggregation-bootstrap-port $PD_BOOTSTRAP_PORT"
        ;;
    decode)
        export TP_SIZE=$PD_TP
        export DP_SIZE=$PD_DP
        export EP_SIZE=$EP_SIZE_PD
        export PORT=$PD_DECODE_PORT
        # Single-host, same reason as the prefill role above.
        export FORCE_SINGLE_HOST=${FORCE_SINGLE_HOST:-1}
        # No bootstrap port of its own: the decode engine learns the prefill's
        # bootstrap address from the router-injected request.
        ROLE_ARGS="--disaggregation-mode decode \
--disaggregation-transfer-backend tpu"
        ;;
    *)
        echo "ERROR: ROLE must be colocated|prefill|decode, got '$ROLE'" >&2
        exit 2
        ;;
esac

# sglang divides --chunked-prefill-size by dp_size under DP attention, so
# pre-multiply to land on CHUNKED_PREFILL_PER_RANK per rank.
export CHUNKED_PREFILL_SIZE=$((CHUNKED_PREFILL_PER_RANK * DP_SIZE))
export MAX_PREFILL_TOKENS=$CHUNKED_PREFILL_SIZE

# --max-running-requests is divided the same way -- pool_configurator.py does
#   num_reqs = max_running_requests // attn_dp_size
# so the value passed must be pre-multiplied to land on MAX_RUNNING per rank.
# Without this, any sweep whose largest concurrency is below dp_size floors to
# zero and EVERY scheduler dies with "max_running_request is zero" after the
# weights are already loaded: PROFILE=smoke sweeps {1,2} against dp=4, so 2//4=0.
export MAX_RUNNING=$((MAX_RUNNING * DP_SIZE))

export MODEL QUANTIZATION KV_CACHE_DTYPE ATTENTION_BACKEND PAGE_SIZE
export CONTEXT_LEN MEM_FRACTION_STATIC MAX_RUNNING TORCH_COMPILE_MAX_BS
export ENABLE_METRICS ENABLE_CACHE_REPORT KV_OFFLOAD HICACHE_RATIO
export JSON_MODEL_OVERRIDE
export HOST=${HOST:-0.0.0.0}
export ENABLE_TORCH_COMPILE=${ENABLE_TORCH_COMPILE:-1}
# Prefix caching is the feature under test, so the radix cache must stay on --
# launch_server.sh disables it by default for synthetic-prefix throughput runs.
# HiCache is layered on the radix tree, so KV offload needs this too.
export DISABLE_RADIX_CACHE=0
export HICACHE_IO_BACKEND=torch_tpu
export HICACHE_MEM_LAYOUT=page_first_fused

# Advertise this pod's cluster IP so the peer engine can reach the bootstrap
# server and the KV sockets (hostname -i may list several; take the first).
if [[ "$ROLE" != "colocated" ]]; then
    export SGLANG_HOST_IP=${SGLANG_HOST_IP:-$(hostname -i | awk '{print $1}')}
fi

export EXTRA_ARGS="$ROLE_ARGS ${EXTRA_ARGS:-}"

log_config "$ROLE"

LOG_FILE=${LOG_FILE:-}
if [[ -n "$LOG_FILE" ]]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    log_config "$ROLE" >>"$LOG_FILE"
    # PIPESTATUS is not preserved through tee, so use a process substitution and
    # keep the server's own exit status as this script's.
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

exec "$TPU_BENCH_DIR/launch_server.sh"
