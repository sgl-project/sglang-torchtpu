#!/bin/bash
# Launch the GLM-5.3-Flash AgentX server of one point on one tpu7x 2x2x1 host (tp8), as the
# measurement harness did. Ready when the log prints "The server is fired up and ready to roll".
# usage: launch_agentx.sh <1|4|8|16|16-kda1400|32> [--dry-run]
#   CACHE_ROOT=<dir>     compile caches and TMPDIR, on a PVC (default /cache/agentx)
#   AGENTX_DUMMY=1       random weights: the measured config exactly (default: real weights)
#   AGENTX_HALF_SLICE=1  one host of a 2x2x2 slice run alone, as the c4-c32 points ran
#   --dry-run            print the resolved command and server args; start nothing
set -euo pipefail
POINT=${1:-}
case "$POINT" in
  1 | 4 | 8 | 16 | 32) RECIPE=agentx_cc$POINT ;;
  16-kda1400) RECIPE=agentx_cc16_kda1400 ;;
  *) echo "usage: $0 <1|4|8|16|16-kda1400|32> [--dry-run]" >&2; exit 2 ;;
esac
case "${2:-}" in
  "") DRY_RUN= ;;
  --dry-run) DRY_RUN=--dry-run ;;
  *) echo "unknown argument: $2" >&2; exit 2 ;;
esac
[ $# -le 2 ] || { echo "too many arguments" >&2; exit 2; }

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.." && pwd)
CR=${CACHE_ROOT:-/cache/agentx}/c$POINT
# The branch's sglang and sglang_tpu, not the image's.
export PYTHONSAFEPATH=1 PYTHONPATH=$ROOT/python:$ROOT PYTHONFAULTHANDLER=1
export TMPDIR=$CR/tmp PROMETHEUS_MULTIPROC_DIR=$CR/prom SGLANG_CACHE_DIR=$CR/sglang
export TORCHINDUCTOR_CACHE_DIR=$CR/inductor TORCH_TPU_TIER3_ROOT=$CR/tier3
export TORCH_TPU_TIER2_NAME=agentx_c$POINT
if [ "${AGENTX_HALF_SLICE:-0}" = 1 ]; then
  export FORCE_SINGLE_HOST=1 TPU_TOPOLOGY=2x2x1 TPU_HOST_BOUNDS=1,1,1 HOST_BOUNDS=1,1,1 \
    TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_WORKER_HOSTNAMES=localhost \
    TPU_WORKER_ID=0 TPU_PROCESS_ADDRESSES=localhost:8471 TPU_ACCELERATOR_TYPE=tpu7x-8
fi
ulimit -c 0 # a core dump of one rank fills the disk
if [ -z "$DRY_RUN" ]; then
  rm -rf "$PROMETHEUS_MULTIPROC_DIR" # stale multiprocess metrics from an earlier boot
  mkdir -p "$TMPDIR" "$PROMETHEUS_MULTIPROC_DIR" "$SGLANG_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
    "$TORCH_TPU_TIER3_ROOT" "$HOME/.cache"
  if [ -d "$HOME/.cache/sglang" ] && [ ! -L "$HOME/.cache/sglang" ]; then
    echo "$HOME/.cache/sglang is a directory; move it so it can point at $SGLANG_CACHE_DIR" >&2
    exit 1
  fi
  ln -sfn "$SGLANG_CACHE_DIR" "$HOME/.cache/sglang"
fi
cd "$ROOT"
WEIGHTS=real
[ "${AGENTX_DUMMY:-0}" = 1 ] && WEIGHTS=dummy
echo "[launch_agentx] point c$POINT: $RECIPE.py, caches $CR, $WEIGHTS weights" >&2
exec python3 -u -m benchmark.tpu.launch_server \
  --config "recipe/configs/glm_5_3_flash/experimental/agentx/$RECIPE.py" $DRY_RUN
