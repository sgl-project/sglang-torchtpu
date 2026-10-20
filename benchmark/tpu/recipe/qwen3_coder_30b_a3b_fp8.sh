#!/usr/bin/env bash
# Single-host launch recipe: Qwen3-Coder-30B-A3B-Instruct-FP8 (tp=8, ep=8) on one v7x
# host. This script IS the model+config spec; the nightly runs exactly it, so a manual
# repro of a CI failure is `./qwen3_coder_30b_a3b_fp8.sh` and nothing else.
#
# MODEL is an HF repo id rather than a /weights path: this config predates the shared
# model-weights-filestore share and pulls from the HF cache. Point MODEL at /weights/<dir>
# once the checkpoint is on the share.
#
# CONTEXT_LEN 8192 with a 262144-token KV pool sizes the server for the lm-eval MMLU
# sweep (test/lm_eval_configs/Qwen3-Coder-30B-A3B-Instruct-FP8-TPU.yaml) at
# MAX_RUNNING=64; mem_fraction 0.30 is what leaves room for that pool at tp=8.
set -euo pipefail
cd "$(dirname "$0")/.."   # recipe/ -> benchmark/tpu
exec env \
  MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8}" \
  TP_SIZE="${TP_SIZE:-8}" \
  EP_SIZE="${EP_SIZE:-8}" \
  PAGE_SIZE="${PAGE_SIZE:-128}" \
  CONTEXT_LEN="${CONTEXT_LEN:-8192}" \
  CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}" \
  MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-4096}" \
  MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-262144}" \
  MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.30}" \
  MAX_RUNNING="${MAX_RUNNING:-64}" \
  TORCH_COMPILE_MAX_BS="${TORCH_COMPILE_MAX_BS:-64}" \
  ENABLE_PROFILER_ENDPOINT="${ENABLE_PROFILER_ENDPOINT:-0}" \
  bash launch_server.sh
