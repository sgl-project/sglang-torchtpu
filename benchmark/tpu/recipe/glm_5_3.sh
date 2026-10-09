#!/usr/bin/env bash
# GLM-5.3 text reference launch with sparse attention (DSA): the indexer picks
# the keys and the sparse MLA kernels attend over them.
# TODO(sglang-torchtpu#1072): allow a context above 2048 tokens.
set -euo pipefail
cd "$(dirname "$0")/.."
glm_context=${CONTEXT_LEN:-2048}
if (( glm_context < 1 || glm_context > 2048 )); then
  echo 'GLM-5.3 reference recipe requires 1 <= CONTEXT_LEN <= 2048.' >&2
  exit 2
fi
exec env \
  MODEL="${MODEL:-/weights/GLM-5.3}" \
  TP_SIZE="${TP_SIZE:-8}" \
  EP_SIZE="${EP_SIZE:-8}" \
  ATTENTION_BACKEND=pallas_tpu_mla_sparse \
  CONTEXT_LEN="$glm_context" \
  CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-128}" \
  MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-128}" \
  MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-4096}" \
  PAGE_SIZE="${PAGE_SIZE:-128}" \
  MAX_RUNNING="${MAX_RUNNING:-2}" \
  TORCH_COMPILE_MAX_BS="${TORCH_COMPILE_MAX_BS:-2}" \
  TPU_COMPILE_TOKENS="${TPU_COMPILE_TOKENS:-128}" \
  TPU_COMPILE_BS="${TPU_COMPILE_BS:-1 2}" \
  MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.95}" \
  KV_CACHE_DTYPE=fp8_e4m3 \
  DISABLE_RADIX_CACHE=1 \
  DISABLE_OVERLAP_SCHEDULE=1 \
  ENABLE_PROFILER_ENDPOINT=0 \
  EXTRA_ARGS="--watchdog-timeout 1800 --disable-shared-experts-fusion ${EXTRA_ARGS:-}" \
  bash launch_server.sh
