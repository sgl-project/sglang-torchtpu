#!/usr/bin/env bash
# SINGLE-HOST Reflection Beam-1.0-501B-A23B-FP8 on 1x tpu7x-8 (2x2x1, 8 devices).
#   tp=8, ep=8, pure TP attention (48 q / 8 kv heads -> 6 q / 1 kv per rank).
#   Weights ~503 GB compressed-tensors FP8 (per-channel weights, per-token
#   dynamic activations) -> ~63 GB/rank of ~101.7 GB HBM.
#
#   ./launch_beam_fp8.sh       # wait for "The server is fired up and ready to roll!"
#
# Checkpoint location, first match wins:
#   BEAM_LOCAL_MODEL=/mnt/disks/local/Beam-1.0-501B-A23B-FP8   # local-disk copy (preferred)
#   MODEL=/gcs/Beam-1.0-501B-A23B-FP8                          # gcsfuse mount (default)
#
# Override any knob via env, e.g.
#   CONTEXT_LEN=32768 MAX_RUNNING=16 ./launch_beam_fp8.sh
#   KV_CACHE_DTYPE=fp8_e4m3 ./launch_beam_fp8.sh       # after bf16 KV accuracy is signed off
# For a weightless plumbing/compile smoke, add EXTRA_ARGS="--load-format dummy".
# Shared env (PYTHON, LIBTPU flags, ...) comes from ../../common_env.sh, sourced
# by launch_server.sh.
set -uo pipefail
cd "$(dirname "$0")/../.."   # -> benchmark/tpu (launch_server.sh + common_env.sh)

if [[ -n "${BEAM_LOCAL_MODEL:-}" && -f "${BEAM_LOCAL_MODEL}/config.json" ]]; then
    MODEL="$BEAM_LOCAL_MODEL"
else
    MODEL="${MODEL:-/gcs/Beam-1.0-501B-A23B-FP8}"
fi

CHUNK="${CHUNKED_PREFILL_SIZE:-4096}"
MAX_RUNNING="${MAX_RUNNING:-64}"

# Checkpoint scales are BF16 [N,1]; keep the MoE expert scales in FP32.
export SGLANG_TPU_FP8_MOE_FP32_SCALES="${SGLANG_TPU_FP8_MOE_FP32_SCALES:-1}"
# Validated on tpu7x 2x2x1 with the router on its torch reference: the Pallas
# router gives wrong prefill output inside the 4096-token x 64-request compiled
# graph (correct standalone). Set to "" to enable it once fixed on-chip.
export SGLANG_DEBUG_TPU_BEAM_KERNELS_SKIP="${SGLANG_DEBUG_TPU_BEAM_KERNELS_SKIP-topk}"

# --weight-loader-disable-mmap: gcsfuse mmap SIGBUSes on large shards, and it
#   enables EP-local expert reads (each rank reads only its 32 of 256 experts).
# QUANTIZATION is empty on purpose: the method (compressed-tensors) comes from
#   the checkpoint's quantization_config; launch_server.sh would otherwise pass
#   --quantization fp8.
MODEL="$MODEL" \
TP_SIZE=8 EP_SIZE=8 \
QUANTIZATION="" \
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bf16}" \
ATTENTION_BACKEND="${ATTENTION_BACKEND:-pallas_tpu_rpa_v3}" \
PAGE_SIZE="${PAGE_SIZE:-128}" \
CONTEXT_LEN="${CONTEXT_LEN:-12288}" \
MAX_RUNNING="$MAX_RUNNING" \
TORCH_COMPILE_MAX_BS="${TORCH_COMPILE_MAX_BS:-$MAX_RUNNING}" \
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}" \
CHUNKED_PREFILL_SIZE="$CHUNK" MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-$CHUNK}" \
TPU_COMPILE_TOKENS="${TPU_COMPILE_TOKENS:-64 128 256 512 1024 2048 4096}" \
TPU_COMPILE_BS="${TPU_COMPILE_BS:-1 2 4 8 16 32 64}" \
EXTRA_ARGS="--weight-loader-disable-mmap --weight-loader-drop-cache-after-load --reasoning-parser beam --tool-call-parser beam ${EXTRA_ARGS:-}" \
exec ./launch_server.sh "$@"
