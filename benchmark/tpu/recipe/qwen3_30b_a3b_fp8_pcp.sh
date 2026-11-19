#!/usr/bin/env bash
# Single-host launch recipe: Qwen3-30B-A3B-FP8 with prefill context parallelism on
# one host (tp=8). Attention CP 8 shards each prefill over the TP group; MoE DP 8
# keeps every expert on every core, so each core routes only its own tokens.
# ATTN_CP_SIZE=4 runs PCP4 x attention TP 2 (MoE DP 4, MoE TP 2).
#
# Prefill CP runs on the batched RPA backend (page size >= 256) and samples with the
# pytorch backend. --moe-runner-backend pallas_tpu is the MoE runner this
# checkpoint is validated with.
set -euo pipefail
cd "$(dirname "$0")/.."   # recipe/ -> benchmark/tpu
exec env \
  MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}" \
  TP_SIZE=8 \
  ATTENTION_BACKEND=pallas_tpu_rpa_batched \
  PAGE_SIZE="${PAGE_SIZE:-256}" \
  CONTEXT_LEN="${CONTEXT_LEN:-8192}" \
  CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}" \
  MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-8192}" \
  MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-131072}" \
  MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.6}" \
  EXTRA_ARGS="--attn-cp-size ${ATTN_CP_SIZE:-8} --enable-prefill-cp --cp-strategy zigzag \
--moe-dp-size ${ATTN_CP_SIZE:-8} --moe-runner-backend pallas_tpu --sampling-backend pytorch \
--watchdog-timeout 3600${EXTRA_ARGS:+ $EXTRA_ARGS}" \
  bash launch_server.sh
