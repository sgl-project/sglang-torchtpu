#!/bin/bash
# run4h.sh <tag> [app args] [-- anchor args]: this host's process of the multi-host standalone server; run it on
# every host (e.g. `c4 bg <h> 'bash .../server/run4h.sh tag ...'` for h = 0..3). HTTP on process 0 (TPU_WORKER_ID 0).
# Log: $OUT_ROOT/<tag>/host<id>.log. Env: TREE, COORD_PORT (default 8477), OUT_ROOT, PY (python),
# TMPDIR / XDG_CACHE_HOME / HF_HOME / TPU_LOG_DIR (defaults under /dev/shm).
set -u
TAG=${1:?usage: run4h.sh <tag> [server args] [-- anchor args]}; shift
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREE=${TREE:-$(cd "$HERE/../../.." && pwd)}
OUT_ROOT=${OUT_ROOT:-$TREE/out/k3mk_server}
: "${TPU_WORKER_ID:?TPU_WORKER_ID is not set (not a TPU worker?)}"
: "${TPU_WORKER_HOSTNAMES:?TPU_WORKER_HOSTNAMES is not set}"
NPROC=$(awk -F, '{print NF}' <<< "$TPU_WORKER_HOSTNAMES")
COORD="$(cut -d, -f1 <<< "$TPU_WORKER_HOSTNAMES"):${COORD_PORT:-8477}"
H=$TPU_WORKER_ID
mkdir -p "$OUT_ROOT/$TAG"
LOG="$OUT_ROOT/$TAG/host$H.log"
ulimit -c 0
export TMPDIR=${TMPDIR:-/dev/shm/mk_tmp} XDG_CACHE_HOME=${XDG_CACHE_HOME:-/dev/shm/mk_cache} HF_HOME=${HF_HOME:-/dev/shm/mk_hf} TPU_LOG_DIR=${TPU_LOG_DIR:-/dev/shm/mk_tpulog}
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TPU_LOG_DIR"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONPATH="$TREE/python:$TREE"
cd "$TREE" || exit 1
# Split app args from anchor args: everything after "--" goes to the anchor parser.
APP=(); REST=(); seen=0
for a in "$@"; do if [ $seen = 0 ] && [ "$a" = "--" ]; then seen=1; elif [ $seen = 0 ]; then APP+=("$a"); else REST+=("$a"); fi; done
echo "run4h: host $H/$NPROC tag $TAG tree $TREE ($(git -C "$TREE" rev-parse --short HEAD)) coordinator $COORD args $*" > "$LOG"
setsid nohup ${PY:-python3} -m sglang_tpu.k3_megakernel.server.app "${APP[@]}" -- \
  --coordinator "$COORD" --num-processes "$NPROC" --process-id "$H" "${REST[@]}" >> "$LOG" 2>&1 < /dev/null &
echo "host$H started $TAG (pid $!), log $LOG"
