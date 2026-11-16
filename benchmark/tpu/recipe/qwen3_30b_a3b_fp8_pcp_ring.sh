#!/usr/bin/env bash
# Single-host launch recipe: Qwen3-30B-A3B-FP8 with ring-mode prefill context
# parallelism on one host (tp=8). Like qwen3_30b_a3b_fp8_pcp.sh, attention CP
# (ATTN_CP_SIZE, default 8) shards each prefill over the attention-CP group and
# MoE DP = ATTN_CP_SIZE runs the MoE on each rank's own tokens.
# SGLANG_TPU_PCP_MODE=ring changes where the KV cache lives: it is striped across
# the ranks of each attention-CP group by page (--dcp-size is set to
# ATTN_CP_SIZE at start-up), a zigzag prefill attends the cached prefix through
# the fused ring kernel, and decode runs the decode context-parallel path over
# the same stripes.
#
# ATTN_CP_SIZE=4 is PCP4 x attention TP 2: each core keeps 2 of Qwen3's 4 KV
# heads (an fp8 cache packs 4 rows per word, so 1 head per core would waste
# half the pool), and the ring runs along the CP axis of a (4, 2) mesh.
# With ATTN_CP_SIZE < 8 the MoE runs MoE DP x EP: EP_SIZE defaults to
# 8 / ATTN_CP_SIZE, so each core holds whole experts (override with EP_SIZE).
#
# MAX_TOTAL_TOKENS is the KV pool of ONE core; with ATTN_CP_SIZE stripes the
# server holds ATTN_CP_SIZE times as many tokens as one core's pool.
# CHUNKED_PREFILL_SIZE must be at least PAGE_SIZE x ATTN_CP_SIZE.
# TPU_COMPILE_TOKENS buckets used by context-parallel batches must be multiples
# of 2 x PAGE_SIZE.
set -euo pipefail
cd "$(dirname "$0")/.."   # recipe/ -> benchmark/tpu
if (( ${ATTN_CP_SIZE:-8} < 8 )); then
  EP_SIZE="${EP_SIZE:-$(( 8 / ${ATTN_CP_SIZE:-8} ))}"
fi
exec env \
  SGLANG_TPU_PCP_MODE=ring \
  MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}" \
  TP_SIZE=8 \
  EP_SIZE="${EP_SIZE:-}" \
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
