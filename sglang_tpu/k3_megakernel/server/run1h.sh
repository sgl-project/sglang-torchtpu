#!/bin/bash
# run1h.sh <tag> [app args] [-- anchor args]: single-host standalone server (TP8, this host's 4 chips).
# Log: $OUT_ROOT/<tag>/server.log. Env: TREE (default: the tree holding this script), OUT_ROOT, PY (python),
# TMPDIR / XDG_CACHE_HOME / HF_HOME / TPU_LOG_DIR (defaults under /dev/shm).
TAG=${1:?usage: run1h.sh <tag> [server args] [-- anchor args]}; shift
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREE=${TREE:-$(cd "$HERE/../../.." && pwd)}
OUT_ROOT=${OUT_ROOT:-$TREE/out/k3mk_server}
# One host of a multi-host tpu7x slice run as an independent 2x2x1 slice.
ulimit -c 0
export TPU_WORKER_HOSTNAMES=localhost TPU_WORKER_ID=0 TPU_TOPOLOGY=2x2x1
unset TORCH_TPU_SLICEBUILDER_ADDRESSES TORCH_TPU_TOPOLOGY
export TPU_VISIBLE_CHIPS=0,1,2,3 TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_HOST_BOUNDS=1,1,1
export TMPDIR=${TMPDIR:-/dev/shm/mk_tmp} XDG_CACHE_HOME=${XDG_CACHE_HOME:-/dev/shm/mk_cache} HF_HOME=${HF_HOME:-/dev/shm/mk_hf} TPU_LOG_DIR=${TPU_LOG_DIR:-/dev/shm/mk_tpulog}
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TPU_LOG_DIR"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONPATH="$TREE/python:$TREE"
cd "$TREE" || exit 1
mkdir -p "$OUT_ROOT/$TAG"
LOG="$OUT_ROOT/$TAG/server.log"
echo "run1h: tag $TAG tree $TREE ($(git -C "$TREE" rev-parse --short HEAD)) args $*" > "$LOG"
setsid nohup ${PY:-python3} -m sglang_tpu.k3_megakernel.server.app "$@" >> "$LOG" 2>&1 < /dev/null &
echo "started $TAG pid $! log $LOG"
