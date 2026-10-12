#!/usr/bin/env bash
# Single-host launch recipe: GPT-OSS-20B with prefill context parallelism on one
# host (tp=8). Attention CP 8 shards each prefill over the TP group; MoE DP 8
# keeps every expert on every core, so each core routes only its own tokens.
#
# This is the PR gate's PCP configuration. It mirrors qwen3_30b_a3b_fp8_pcp.sh,
# which it replaces on the gate: PCP is an attention feature and does not need a
# 30B checkpoint to exercise, while the 30B's weight load plus its share of the
# compile grid made it the most expensive launch in the e2e suite. The 30B
# recipe stays for the nightly and for benchmarking.
#
# GPT-OSS specifics carried over from gpt_oss_120b/server.sh: MXFP4 weights ship
# their own quantization_config, so no online quantization; the SparseCore
# ragged-gather MoE path cannot be lowered for this architecture; and the model
# needs attention sinks, which the batched RPA backend implements.
set -uo pipefail
cd "$(dirname "$0")/.."   # recipe/ -> benchmark/tpu
exec env \
  MODEL="${MODEL:-openai/gpt-oss-20b}" \
  TP_SIZE=8 \
  QUANTIZATION=unquant \
  KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}" \
  SGLANG_TPU_FUSED_MOE_SC_GATHER="${SGLANG_TPU_FUSED_MOE_SC_GATHER:-0}" \
  ATTENTION_BACKEND=pallas_tpu_rpa_batched \
  PAGE_SIZE="${PAGE_SIZE:-256}" \
  CONTEXT_LEN="${CONTEXT_LEN:-8192}" \
  CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}" \
  MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-8192}" \
  MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-131072}" \
  MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.6}" \
  EXTRA_ARGS="--attn-cp-size 8 --enable-prefill-cp --cp-strategy zigzag \
--moe-dp-size 8 --moe-runner-backend pallas_tpu --sampling-backend pytorch \
--watchdog-timeout 3600${EXTRA_ARGS:+ $EXTRA_ARGS}" \
  bash launch_server.sh
