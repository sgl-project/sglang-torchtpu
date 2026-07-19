#!/usr/bin/env bash
# Sample usage: ISL=8192 OSL=1024 CONCUR=64 ./server.sh
set -uo pipefail

ISL=${ISL:-8192}
OSL=${OSL:-1024}
CONCUR=${CONCUR:-64}
export CONTEXT_LEN=${CONTEXT_LEN:-$((ISL + OSL))}
# KV pool sized so every in-flight request can hold a full context.
export MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-$((CONCUR * CONTEXT_LEN))}
CHUNK=${CHUNK:-$ISL}

export MODEL=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
export TP_SIZE=8
export EP_SIZE=8
export QUANTIZATION=fp8
export KV_CACHE_DTYPE=fp8_e4m3
export ATTENTION_BACKEND=pallas_tpu_rpa_v3
export PAGE_SIZE=128
export MEM_FRACTION_STATIC=0.8
export MAX_RUNNING=$CONCUR
export CHUNKED_PREFILL_SIZE=$CHUNK
export MAX_PREFILL_TOKENS=$CHUNK
export ENABLE_TORCH_COMPILE=1
export TORCH_COMPILE_MAX_BS=$CONCUR

export EXTRA_ARGS="--tpu-fp8-runtime channelwise --sampling-defaults openai${EXTRA_ARGS:+ $EXTRA_ARGS}"

cd "$(dirname "$0")/../.."
exec ./launch_server.sh "$@"
