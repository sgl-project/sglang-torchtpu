#!/usr/bin/env bash
# Qwen3-VL-8B-Instruct (bf16 dense decoder + ViT) on one tpu7x host — 4 chips x 2 cores = tp8.
#
# Sample usage:
#   ./server.sh                                                  # 8k/1k serving shape
#   EXTRA_ARGS="--speculative-algorithm EAGLE3 ..." ./server.sh  # + spec decoding
set -euo pipefail

# HF id by default; CI tests supply the /weights path explicitly.
export MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
export TP_SIZE=8

# bf16 checkpoint. "unquant" is the explicit opt-out, not merely an unset: it
# also sets _quantization_explicitly_unset, so the model overrides cannot
# re-detect a quant method. Empty would only drop launch_server.sh's fp8.
export QUANTIZATION=unquant

export PAGE_SIZE=${PAGE_SIZE:-128}
export CONTEXT_LEN=${CONTEXT_LEN:-9216}
export CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
export MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-8192}
export MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-262144}

export TORCH_COMPILE_MAX_BS=${TORCH_COMPILE_MAX_BS:-64}
# 8B bf16 is ~2 GB per rank, so the budget is dominated by graph temporaries:
# the ViT and the decoder each capture their own.
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.30}

# max_pixels caps one image at ~306 vision tokens (patch 16 + 2x2 merge = 1024
# px per token). Uncapped, a single MMMU-Pro page exceeds the whole 8k chunk.
MM_PROCESS_CONFIG='{"image":{"min_pixels":65536,"max_pixels":313600}}'
# --watchdog-timeout: the EXTEND/decode graph captures outlast the 300s default
# and trip the watchdog into repeated stack dumps mid-compile.
export EXTRA_ARGS="--trust-remote-code --watchdog-timeout 3600 \
--mm-process-config $MM_PROCESS_CONFIG${EXTRA_ARGS:+ $EXTRA_ARGS}"

cd "$(dirname "$0")/../.."
exec ./launch_server.sh "$@"
