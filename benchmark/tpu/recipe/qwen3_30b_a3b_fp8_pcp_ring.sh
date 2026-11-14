#!/usr/bin/env bash
# Single-host launch recipe: Qwen3-30B-A3B-FP8 with ring-mode prefill context
# parallelism on one host (tp=8). Like qwen3_30b_a3b_fp8_pcp.sh, attention CP 8
# shards each prefill over the TP group and MoE DP 8 keeps every expert on every
# core. SGLANG_TPU_PCP_MODE=ring changes where the KV cache lives: it is striped
# across the eight cores by page (--dcp-size 8), a zigzag prefill attends the
# cached prefix through the fused ring kernel, and decode runs the decode
# context-parallel path over the same stripes.
#
# MAX_TOTAL_TOKENS is the KV pool of ONE core; with eight stripes the server
# holds eight times as many tokens as the gather-mode recipe at the same value.
# CHUNKED_PREFILL_SIZE must be at least PAGE_SIZE x 8. TPU_COMPILE_TOKENS
# buckets used by context-parallel batches must be multiples of 2 x PAGE_SIZE.
set -euo pipefail
cd "$(dirname "$0")/.."   # recipe/ -> benchmark/tpu
exec env \
  SGLANG_TPU_PCP_MODE=ring \
  MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}" \
  TP_SIZE=8 \
  ATTENTION_BACKEND=pallas_tpu_rpa_batched \
  PAGE_SIZE="${PAGE_SIZE:-256}" \
  CONTEXT_LEN="${CONTEXT_LEN:-8192}" \
  CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}" \
  MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-8192}" \
  MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-131072}" \
  MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.6}" \
  EXTRA_ARGS="--attn-cp-size 8 --enable-prefill-cp --cp-strategy zigzag \
--dcp-size 8 \
--moe-dp-size 8 --moe-runner-backend pallas_tpu --sampling-backend pytorch \
--watchdog-timeout 3600${EXTRA_ARGS:+ $EXTRA_ARGS}" \
  bash launch_server.sh
