#!/usr/bin/env bash
# Multi-host GPT-OSS-120B DFlash validation server for the TPU nightly.
#
# The JobSet runs this identical command on both hosts. launch_server.sh reads
# GKE-owned TPU_WORKER_ID / TPU_WORKER_HOSTNAMES and builds one no-Ray global
# process group: tp=16, ep=16 on a 2x2x2 v7x slice. The GSM8K client drives 64
# concurrent requests.
#
# DP_SIZE is pinned empty, not left overridable: _handle_dflash rejects
# enable_dp_attention outright, so any DP_SIZE > 1 dies in prepare_server_args.
# MAX_TOTAL_TOKENS caps one attention worker's pool, so collapsing dp=4 into a
# single tp=16 worker needs 4x the old 32768 to keep the per-chip KV footprint.
#
# Keep this recipe focused on speculative decoding. Prefix-cache, HiCache, and
# P/D combinations have separate tests and should not obscure this gate.
set -euo pipefail

cd "$(dirname "$0")/.."  # recipe/ -> benchmark/tpu

model=${MODEL:-/weights/gpt-oss-120b}
# The draft sits next to the target in the external cluster's gcsfuse weight bucket.
draft_model=${DRAFT_MODEL:-$(dirname "$model")/gpt-oss-120b-DFlash}
served_model_name=${SERVED_MODEL_NAME:-gpt-oss-120b}

spec_args=(
  --reasoning-parser gpt-oss
  --served-model-name "$served_model_name"
  --speculative-algorithm DFLASH
  --speculative-draft-model-path "$draft_model"
  --speculative-dflash-block-size 6
  --tpu-compile-tracer dynamo
)

exec env \
  MODEL="$model" \
  TP_SIZE="${TP_SIZE:-16}" \
  EP_SIZE="${EP_SIZE:-16}" \
  DP_SIZE= \
  ISL="${ISL:-4096}" \
  OSL="${OSL:-512}" \
  CONC="${CONC:-64}" \
  MAX_RUNNING="${MAX_RUNNING:-64}" \
  CONTEXT_LEN="${CONTEXT_LEN:-8192}" \
  CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}" \
  MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-4096}" \
  MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-131072}" \
  TPU_COMPILE_TOKENS="${TPU_COMPILE_TOKENS:-512 1024 2048 4096}" \
  TPU_COMPILE_BS="${TPU_COMPILE_BS:-1 2 4 8 16}" \
  PAGE_SIZE="${PAGE_SIZE:-64}" \
  ATTENTION_BACKEND="${ATTENTION_BACKEND:-pallas_tpu_rpa_v3}" \
  MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.6}" \
  QUANTIZATION= \
  KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}" \
  DISABLE_RADIX_CACHE=1 \
  ENABLE_TORCH_COMPILE=1 \
  ENABLE_METRICS=1 \
  SGLANG_TPU_FUSED_MOE_SC_GATHER=0 \
  SGLANG_SPEC_TP_SYNC="${SGLANG_SPEC_TP_SYNC:-rng}" \
  SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER="${SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER:-1}" \
  EXTRA_ARGS="${spec_args[*]}${EXTRA_ARGS:+ $EXTRA_ARGS}" \
  bash launch_server.sh "$@"
