#!/usr/bin/env bash
# Gemma-4-31B-it (bf16 dense, hybrid sliding/global attention + ViT) on one tpu7x
# host — 4 chips x 2 cores = tp8.
#
# Sample usage:
#   ./server.sh                                # 8k/1k serving shape
#   MODEL=/weights/gemma-4-31B-it ./server.sh  # CI weights share
set -uo pipefail

export MODEL=${MODEL:-google/gemma-4-31B-it}
export TP_SIZE=8

# bf16 checkpoint; "unquant" also blocks the model overrides from re-detecting a
# quant method (an empty value only drops launch_server.sh's fp8 default).
export QUANTIZATION=unquant
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-bfloat16}

export PAGE_SIZE=${PAGE_SIZE:-128}
export CONTEXT_LEN=${CONTEXT_LEN:-9216}
export CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
export MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-8192}
# Bounds the hybrid pool: 10 global layers at head_dim 512, 50 sliding at 256.
export MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-131072}

export TORCH_COMPILE_MAX_BS=${TORCH_COMPILE_MAX_BS:-64}
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.35}

# The multimodal wrapper runs eagerly around the compiled text model; bump the
# positions out of place rather than mutating the batch's tensor.
export SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION=1

# --watchdog-timeout: EXTEND/decode graph captures outlast the 300s default.
export EXTRA_ARGS="--watchdog-timeout 3600${EXTRA_ARGS:+ $EXTRA_ARGS}"

cd "$(dirname "$0")/../.."
exec ./launch_server.sh "$@"
