#!/bin/bash
# run4h.sh <tag> [anchor.py args...]: start this host's process of a multi-host anchor run.
# Run it on every host of the slice at about the same time (e.g. `c4 bg <h> 'bash .../run4h.sh tag ...'`
# for h = 0..3). Process id = TPU_WORKER_ID; the coordinator is the first entry of
# TPU_WORKER_HOSTNAMES. Log: $OUT_ROOT/<tag>/host<id>.log; records: $OUT_ROOT/<tag>/anchor.jsonl.
# Env overrides: TREE (code tree, default: the tree holding this script), COORD_PORT (default 8476),
# OUT_ROOT (default $TREE/out/k3mk_harness), PY (default python3),
# MAIN (bench module: anchor, or serve for the resident worker; then pass --queue <dir>).
set -u
TAG=${1:?usage: run4h.sh <tag> [anchor args...]}
shift
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREE=${TREE:-$(cd "$HERE/../../../.." && pwd)}
OUT_ROOT=${OUT_ROOT:-$TREE/out/k3mk_harness}
: "${TPU_WORKER_ID:?TPU_WORKER_ID is not set (not a TPU worker?)}"
: "${TPU_WORKER_HOSTNAMES:?TPU_WORKER_HOSTNAMES is not set}"
NPROC=$(awk -F, '{print NF}' <<< "$TPU_WORKER_HOSTNAMES")
COORD="$(cut -d, -f1 <<< "$TPU_WORKER_HOSTNAMES"):${COORD_PORT:-8476}"
H=$TPU_WORKER_ID
mkdir -p "$OUT_ROOT/$TAG"
LOG="$OUT_ROOT/$TAG/host$H.log"

ulimit -c 0
export TMPDIR=/dev/shm/mk_tmp XDG_CACHE_HOME=/dev/shm/mk_cache HF_HOME=/dev/shm/mk_hf TPU_LOG_DIR=/dev/shm/mk_tpulog
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TPU_LOG_DIR"
# The coordination service is gRPC between pods; an HTTP proxy in the environment breaks it.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONPATH="$TREE/python:$TREE:$TREE/test/manual/k3_megakernel"
cd "$TREE" || exit 1

{
  echo "run4h: host $H/$NPROC tag $TAG tree $TREE ($(git -C "$TREE" rev-parse --short HEAD 2>/dev/null)) coordinator $COORD"
  echo "run4h: args $*"
} > "$LOG"
setsid nohup ${PY:-python3} -m bench.${MAIN:-anchor} \
  --coordinator "$COORD" --num-processes "$NPROC" --process-id "$H" \
  --tag "$TAG" --out "$OUT_ROOT/$TAG/anchor.jsonl" "$@" >> "$LOG" 2>&1 < /dev/null &
echo "host$H started $TAG (pid $!), log $LOG"
