#!/usr/bin/env bash
# Launch one engine for the Qwen3-Coder-480B FP8 TPU agentic benchmark.
# Thin wrapper over benchmark/tpu/launch_server.sh: it only translates the
# contract in config.sh into that script's env interface, so there is exactly
# one place where server flags are decided.
#
#   ROLE=colocated ./serve.sh      # tp=16 dp=2 (attn_tp=8), one engine
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
# ...but --max-prefill-tokens is NOT divided the same way. server_args only
# divides chunked_prefill_size ("chunked prefill size is adjusted from X to
# X//dp"); max_prefill_tokens is already documented as per-DP-rank ("largest
# number of tokens a single forward routes through a MoE layer on one (DP)
# rank") and is what sizes the extend precompile buckets. Passing the
# pre-multiplied value therefore built buckets dp_size times too large and the
# XLA compile OOM'd on HBM (99.26G of temporaries vs 94.74G available) after the
# weights and KV pool were already resident.
export MAX_PREFILL_TOKENS=$CHUNKED_PREFILL_PER_RANK

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
# raiden vs torch_tpu: both are TPU KV-offload io backends; both require
# page_first_fused and a KV pool that supports them (rpa_v3's fused pool does
# both). raiden is the actively developed path and routes through
# TpuDMAHostKVPool instead of
# TpuFusedMHATokenToKVPoolHost -- which matters because hiradix_cache only runs
# precompile_kv_offload() for the fused pool. That warmup stages a
# pow2-ROUNDED whole-pool D2H (next_power_of_2(device_blocks)), needing ~2x the
# KV pool in HBM; measured on a 480B tp=16 slice it asked for 31 GB and forced
# mem-fraction-static down to 0.45 to fit. raiden skips it entirely.
export HICACHE_IO_BACKEND
export HICACHE_MEM_LAYOUT=page_first_fused

# The shared recipe is non-speculative by default. Its multi-host CI shim opts
# in by setting DRAFT_MODEL to the draft checkpoint on the weights PVC.
SPEC_ARGS=""
if [[ -n "$DRAFT_MODEL" ]]; then
    SPEC_ARGS="--speculative-algorithm EAGLE3 \
--speculative-draft-model-path $DRAFT_MODEL \
--speculative-num-steps $SPECULATIVE_NUM_STEPS \
--speculative-eagle-topk $SPECULATIVE_EAGLE_TOPK \
--speculative-num-draft-tokens $SPECULATIVE_NUM_DRAFT_TOKENS"
fi

# Advertise this pod's cluster IP so the peer engine can reach the bootstrap
# server and the KV sockets (hostname -i may list several; take the first).
if [[ "$ROLE" != "colocated" ]]; then
    export SGLANG_HOST_IP=${SGLANG_HOST_IP:-$(hostname -i | awk '{print $1}')}
fi

# Release checkpoint pages before allocating HiCache and compiler state in both
# baseline and EAGLE runs; otherwise host memory can run out despite free HBM.
export EXTRA_ARGS="$ROLE_ARGS $SPEC_ARGS --weight-loader-drop-cache-after-load ${EXTRA_ARGS:-}"

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
