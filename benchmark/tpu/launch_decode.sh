#!/usr/bin/env bash
#
# Launch the DECODE engine for Qwen3-30B-A3B P/D disaggregation on TPU.
# Defaults: TP=8, EP=8 (one 8-chip pod). Run this on the *decode* pod (pair with
# launch_prefill.sh on the prefill pod + a load balancer).
#
# The decode learns the prefill's bootstrap address from the LB-injected
# request, so it needs no bootstrap port of its own.
#
# Usage:
#   MODEL=/path/to/Qwen3-30B-A3B ./benchmark/tpu/launch_decode.sh
#
# Env (optional):
#   TP_SIZE=16  EP_SIZE=16  PAGE_SIZE=64  MEM_FRACTION_STATIC=0.92  PORT=30000
#   PYTHON=/cache/work/.venv/bin/python
#   SGLANG_HOST_IP=<pod ip>          # default: auto from `hostname -i`
#   NNODES=1  NODE_RANK=0  DIST_INIT_ADDR=<node0_ip>:<port>   # multi-node TP only
#   EXTRA_ARGS="..."
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MODEL:?set MODEL (HF repo or local snapshot dir)}"
TP_SIZE=${TP_SIZE:-8}
EP_SIZE=${EP_SIZE:-8}
DP_SIZE=${DP_SIZE:-}
ENABLE_DP_ATTENTION=${ENABLE_DP_ATTENTION:-0}
PAGE_SIZE=${PAGE_SIZE:-64}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.92}
PORT=${PORT:-30000}
PYTHON=${PYTHON:-/cache/work/.venv/bin/python}
HOST=0.0.0.0

export SGLANG_HOST_IP=${SGLANG_HOST_IP:-$(hostname -i | awk '{print $1}')}

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
DIST_INIT_ADDR=${DIST_INIT_ADDR:-}
MN_ARGS=""
if [[ "$NNODES" != "1" ]]; then
    : "${DIST_INIT_ADDR:?set DIST_INIT_ADDR=<node0_ip>:<port> for multi-node TP}"
    MN_ARGS="--nnodes $NNODES --node-rank $NODE_RANK --dist-init-addr $DIST_INIT_ADDR"
fi

echo "[decode] model=$MODEL tp=$TP_SIZE ep=$EP_SIZE host_ip=$SGLANG_HOST_IP" \
     "http=$HOST:$PORT nnodes=$NNODES"

# Keep torch.compile ON at tp>1 with fullgraph: materialize=false keeps the c10d
# collectives *inside* the compiled graph (the tpu backend lowers them) rather
# than graph-breaking, which fullgraph=True forbids. Override via env if needed.
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=${TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS:-false}

export MODEL TP_SIZE EP_SIZE DP_SIZE ENABLE_DP_ATTENTION PAGE_SIZE MEM_FRACTION_STATIC PYTHON HOST PORT
export MAX_RUNNING=${MAX_RUNNING:-}
export TORCH_COMPILE_MAX_BS=${TORCH_COMPILE_MAX_BS:-}
export TPU_COMPILE_BS=${TPU_COMPILE_BS:-}
export TPU_COMPILE_TOKENS=${TPU_COMPILE_TOKENS:-}
export EXTRA_ARGS="--ep-size $EP_SIZE \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend tpu \
    $MN_ARGS ${EXTRA_ARGS:-}"

exec "$HERE/launch_server.sh"
