#!/bin/bash
# Launch one GLM-5.3-Flash AgentX point on one tpu7x 2x2x1 host (tp8) from its run JSON, as jobs
# 36d0 (c1-c16), 83cb (c32), ae9c (c36) and e2df (c48) ran them. Ready when the log prints "The
# server is fired up and ready to roll".
# usage: launch_agentx.sh <1|4|8|16|32|36|48> [--dry-run]
#   CACHE_ROOT=<dir>     compile caches and TMPDIR, on a PVC (default /cache/agentx)
#   AGENTX_DEPS=<dir>    pip --target dir: transformers 5.19.0, blake3, timm 1.0.25 (default $CACHE_ROOT/pydeps)
#   AGENTX_HALF_SLICE=1  one host of a 2x2x2 slice run alone, as c1-c36 ran
#   AGENTX_NO_SPMD=1     c1 only: use_spmd_compile false (torch_tpu 0.2.0), as c1 ran; writes the JSON to the cache dir
#   --dry-run            print the resolved env, command and server args; start nothing
# Removed: point 16-kda1400 and AGENTX_DUMMY, which launched the dummy-weight agentx_cc<N>.py recipes.
set -euo pipefail
case "${1:-}" in
  1) RUN=R1b_c1_real_l3_ring_full ;;
  4) RUN=C4u_c4_real_l3_ring_full ;;
  8) RUN=K2ts_c8_real_l3_ring_full ;;
  16) RUN=C16_S2ts_kda1400_bf16_real_l3_ring_full ;;
  32) RUN=R10_c32_mr32_combo_real_l3_ring_full_ixsplit ;;
  36) RUN=R10_c36_mr32_combo_real_l3_ring_full_ixsplit ;; # client CC=36, 32 running
  48) RUN=R9_c48_mr48_real_l3_ring ;;
  *) echo "usage: $0 <1|4|8|16|32|36|48> [--dry-run]" >&2; exit 2 ;;
esac
POINT=$1
case "${2:-}" in
  "") DRY_RUN= ;;
  --dry-run) DRY_RUN=1 ;;
  *) echo "unknown argument: $2" >&2; exit 2 ;;
esac
[ $# -le 2 ] || { echo "too many arguments" >&2; exit 2; }

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../../../../../.." && pwd)
CFG=$HERE/runs/$RUN.json
CR=${CACHE_ROOT:-/cache/agentx}/c$POINT
DEPS=${AGENTX_DEPS:-${CACHE_ROOT:-/cache/agentx}/pydeps}
if [ "${AGENTX_NO_SPMD:-0}" = 1 ]; then
  [ "$POINT" = 1 ] || { echo "AGENTX_NO_SPMD=1 applies to c1 only" >&2; exit 2; }
  mkdir -p "$CR"
  python3 -I -c 'import json, sys; d = json.load(open(sys.argv[1])); d["server_args"]["use_spmd_compile"] = False; open(sys.argv[2], "w").write(json.dumps(d, indent=2) + "\n")' \
    "$CFG" "$CR/${RUN}_nospmd.json"
  CFG=$CR/${RUN}_nospmd.json
fi
[ -d "$DEPS" ] || echo "[launch_agentx] warning: no $DEPS; the images lack transformers 5.19.0, blake3, timm 1.0.25" >&2
unset ENABLE_PROFILER_ENDPOINT SGLANG_TORCH_PROFILER_DIR # no profiler, as measured
# DEPS first, then the branch's sglang and sglang_tpu, not the image's.
export PYTHONSAFEPATH=1 PYTHONPATH=$DEPS:$ROOT/python:$ROOT PYTHONFAULTHANDLER=1 HF_HUB_OFFLINE=1
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
CMD=(python3 -u -m benchmark.tpu.launch_server --config "@$CFG")
if [ -n "$DRY_RUN" ]; then CMD+=(--dry-run); fi
env | grep -E '^(PYTHON|HF_HUB_OFFLINE|TMPDIR|PROMETHEUS|SGLANG_CACHE|TORCH|FORCE_SINGLE|TPU_|HOST_BOUNDS|CHIPS_PER)' |
  sort | sed 's/^/[launch_agentx] env /' >&2
echo "[launch_agentx] c$POINT, in $ROOT: ${CMD[*]}" >&2
exec "${CMD[@]}"
