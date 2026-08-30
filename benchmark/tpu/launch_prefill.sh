#!/usr/bin/env bash
#
# Launch the PREFILL engine for Qwen3-30B-A3B P/D disaggregation on TPU.
# Defaults: TP=8, EP=8 (one 8-chip pod). Run this on the *prefill* pod; run launch_decode.sh
# on the decode pod; then start a load balancer (pd_lb.py) pointing at both.
#
# Usage:
#   MODEL=/path/to/Qwen3-30B-A3B ./benchmark/tpu/launch_prefill.sh
#
# Env (optional):
#   TP_SIZE=16  EP_SIZE=16  PAGE_SIZE=64  MEM_FRACTION_STATIC=0.92
#   PORT=30000  BOOTSTRAP_PORT=8998
#   PYTHON=/cache/work/.venv/bin/python
#   SGLANG_HOST_IP=<pod ip>          # default: auto from `hostname -i`
#   # Multi-node TP (only if ONE engine spans multiple pods; default 1 pod/16 chips):
#   NNODES=1  NODE_RANK=0  DIST_INIT_ADDR=<node0_ip>:<port>
#   # KV-transfer optimization knobs (default on): SGLANG_TPU_KV_CHANNELS, _ASYNC, _STATS_EVERY
#   EXTRA_ARGS="..."                 # any extra launch_server flags
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MODEL:?set MODEL (HF repo or local snapshot dir)}"
TP_SIZE=${TP_SIZE:-8}
EP_SIZE=${EP_SIZE:-8}
PAGE_SIZE=${PAGE_SIZE:-64}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.92}
PORT=${PORT:-30000}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
PYTHON=${PYTHON:-/cache/work/.venv/bin/python}
HOST=0.0.0.0

# Advertise this pod's cluster IP for the bootstrap server + KV ZMQ sockets;
# the decode pod must be able to reach it. (hostname -i may list several; take 1.)
export SGLANG_HOST_IP=${SGLANG_HOST_IP:-$(hostname -i | awk '{print $1}')}

# Multi-node tensor parallelism — only needed when a single engine spans more
# than one pod. For one 16-chip pod, leave NNODES=1.
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
DIST_INIT_ADDR=${DIST_INIT_ADDR:-}
MN_ARGS=""
if [[ "$NNODES" != "1" ]]; then
    : "${DIST_INIT_ADDR:?set DIST_INIT_ADDR=<node0_ip>:<port> for multi-node TP}"
    MN_ARGS="--nnodes $NNODES --node-rank $NODE_RANK --dist-init-addr $DIST_INIT_ADDR"
fi

echo "[prefill] model=$MODEL tp=$TP_SIZE ep=$EP_SIZE host_ip=$SGLANG_HOST_IP" \
     "http=$HOST:$PORT bootstrap=$BOOTSTRAP_PORT nnodes=$NNODES"

# Keep torch.compile ON at tp>1 with fullgraph: materialize=false keeps the c10d
# collectives *inside* the compiled graph (the tpu backend lowers them) rather
# than graph-breaking, which fullgraph=True forbids. Override via env if needed.
export TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=${TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS:-false}

export MODEL TP_SIZE PAGE_SIZE MEM_FRACTION_STATIC PYTHON HOST PORT
export MAX_RUNNING=${MAX_RUNNING:-}
export TORCH_COMPILE_MAX_BS=${TORCH_COMPILE_MAX_BS:-}
export TPU_COMPILE_BS=${TPU_COMPILE_BS:-}
export TPU_COMPILE_TOKENS=${TPU_COMPILE_TOKENS:-}
export EXTRA_ARGS="--ep-size $EP_SIZE \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend tpu \
    --disaggregation-bootstrap-port $BOOTSTRAP_PORT \
    $MN_ARGS ${EXTRA_ARGS:-}"

exec "$HERE/launch_server.sh"
