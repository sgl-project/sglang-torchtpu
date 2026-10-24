#!/usr/bin/env bash
# TWO-HOST Reflection Beam-1.0-501B-A23B (BF16, served as BF16) on a tpu7x 2x2x2
# slice (2 hosts x 8 devices = 16 ranks, one ICI domain).
#   tp=16, ep=8 (default): EP8 x MoE-TP2, i.e. 32 routed experts per rank pair,
#   each expert's intermediate split over the 2 ranks. Beam routes into 4 of 8
#   groups of 32 contiguous experts and a group is exactly one EP8 rank, so the
#   per-step work of a busy group is shared by 2 devices: the busiest device
#   streams ~30% fewer expert bytes at bs 1 (~12% at bs 64) than with ep=16,
#   and prefill imbalance drops from ~1.9x to ~1.5x. Cost: a 2-way reduction
#   on adjacent ranks after the 8-way EP one. EP_SIZE=16 for the A/B.
#   Pure TP attention: 48 q / 8 kv heads -> 3 q heads per rank and each KV head
#   replicated on 2 ranks (QKVParallelLinear + a separate gate GEMM; the packed
#   q|k|v|gate GEMM needs kv_heads >= tp).
#   Weights 1002.66 GB -> ~62.9 GB/rank of ~101.7 GB.
#
# Run the SAME command on every host. launch_server.sh derives NNODES /
# NODE_RANK / RANK0_HOST from GKE's TPU_WORKER_HOSTNAMES / TPU_WORKER_ID (set
# on tpuq and JobSet pods alike) and rendezvouses on host 0:$DIST_PORT; only
# host 0 serves HTTP. On tpuq:
#   tpuq reserve 8 <hours>                    # one 2x2x2 slice
#   kubectl exec -it <pod-host0> -- bash -lc 'STAGE=1 .../launch_beam_bf16_2host.sh'
#   kubectl exec -it <pod-host1> -- bash -lc 'STAGE=1 .../launch_beam_bf16_2host.sh'
#
# Checkpoint. 1.0 TB does not fit in one host's /dev/shm (~919 GB), but each
# host's ranks read only their own experts (EP-local reads), i.e. half of them.
# STAGE=1 first copies this host's slice from $SRC_MODEL into a sparse copy in
# /dev/shm (header + non-expert tensors + experts [128*h, 128*h+128): ~506 GB),
# then serves from it; re-runs skip staged shards. Without STAGE the model
# streams from $MODEL (gcsfuse), ~600 GB per host per start.
#
# Knobs (env): MODEL, SRC_MODEL, STAGE, EP_SIZE (8 or 16),
# CONTEXT_LEN, MAX_RUNNING, MEM_FRACTION_STATIC, KV_CACHE_DTYPE, EXTRA_ARGS, ...
# DFlash speculative decoding is not available here: the drafter's 40 heads do
# not divide over 16 ranks.
set -uo pipefail
cd "$(dirname "$0")/../.."   # -> benchmark/tpu (launch_server.sh + common_env.sh)

# Off GKE, set NNODES=2 NODE_RANK=<0|1> RANK0_HOST=<host 0> by hand.
if [[ -z "${NNODES:-}" && -n "${TPU_WORKER_HOSTNAMES:-}" ]]; then
    NNODES=$(awk -F, '{print NF}' <<< "$TPU_WORKER_HOSTNAMES")
fi
if [[ "${NNODES:-}" != "2" ]]; then
    echo "ERROR: needs a 2-host slice (TPU_WORKER_HOSTNAMES) or NNODES=2; got '${NNODES:-}'" >&2
    exit 2
fi
HOST_ID="${NODE_RANK:-${TPU_WORKER_ID:-0}}"
# CI mounts the shared weights PVC at /weights; tpuq pods have the gcsfuse mount.
if [[ -z "${SRC_MODEL:-}" && -f /weights/Beam-1.0-501B-A23B/config.json ]]; then
    SRC_MODEL=/weights/Beam-1.0-501B-A23B
fi
SRC_MODEL="${SRC_MODEL:-/gcs/Beam-1.0-501B-A23B}"
STAGED_MODEL="${STAGED_MODEL:-/dev/shm/Beam-1.0-501B-A23B}"

if [[ "${STAGE:-0}" == "1" ]]; then
    # Host h's ranks are tp ranks 8h..8h+7, which own routed experts
    # [128h, 128h+128) under both EP8 x MoE-TP2 and EP16.
    PER_HOST=$((256 / NNODES))
    LO=$((HOST_ID * PER_HOST))
    source ./common_env.sh
    "${PYTHON:-python3}" -m sglang_tpu.stage_ep_local_checkpoint \
        --src "$SRC_MODEL" --dst "$STAGED_MODEL" \
        --expert-range "$LO" "$((LO + PER_HOST))" || exit 1
    MODEL="$STAGED_MODEL"
fi
MODEL="${MODEL:-$SRC_MODEL}"

CHUNK="${CHUNKED_PREFILL_SIZE:-4096}"
MAX_RUNNING="${MAX_RUNNING:-32}"

# QUANTIZATION / KV_CACHE_DTYPE explicit: launch_server.sh would otherwise pass
# --quantization fp8 (online per-tensor FP8 of the whole model on device, which
# needs the BF16 weights resident first) and fp8 KV.
# --weight-loader-disable-mmap enables EP-local expert reads (and avoids gcsfuse
# mmap SIGBUS when streaming).
exec env \
  MODEL="$MODEL" \
  NNODES=2 CHIPS_PER_HOST=8 DIST_PORT="${DIST_PORT:-10011}" \
  TP_SIZE=16 EP_SIZE="${EP_SIZE:-8}" \
  QUANTIZATION="" \
  KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bf16}" \
  ATTENTION_BACKEND="${ATTENTION_BACKEND:-pallas_tpu_rpa_v3}" \
  PAGE_SIZE="${PAGE_SIZE:-128}" \
  CONTEXT_LEN="${CONTEXT_LEN:-8192}" \
  MAX_RUNNING="$MAX_RUNNING" \
  TORCH_COMPILE_MAX_BS="${TORCH_COMPILE_MAX_BS:-$MAX_RUNNING}" \
  MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}" \
  CHUNKED_PREFILL_SIZE="$CHUNK" MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-$CHUNK}" \
  TPU_COMPILE_TOKENS="${TPU_COMPILE_TOKENS:-64 128 256 512 1024 2048 4096}" \
  TPU_COMPILE_BS="${TPU_COMPILE_BS:-1 2 4 8 16 32}" \
  EXTRA_ARGS="--weight-loader-disable-mmap ${EXTRA_ARGS:-}" \
  bash launch_server.sh "$@"
