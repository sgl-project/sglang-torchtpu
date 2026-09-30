#!/usr/bin/env bash
# DeepSeek-V3-architecture THROUGHPUT recipe -- ONE tpu7x host, 4 chips x 2 cores = tp8.
# Serves R1-0528 and V3.1 (same 671B blockwise-fp8 backbone, same shapes); the
# caller picks MODEL.
#
# Sample usage:
#   MODEL=/weights/DeepSeek-V3.1 ISL=8192 OSL=1024 CONCUR=16 ./deepseek_v3_perf_tp8.sh
#
# WHY tp8 AND NOT THE tp16 ACCURACY RECIPE: a perf job is one single-host slice.
# The 642 GB fp8 checkpoint is ~80 GB per rank at tp8, of the 101.7 GB a rank
# owns, so the KV pool is what is left over (~175k fp8 tokens, measured
# 2026-09-04) -- that is the ceiling on CONCUR, not the client.
set -uo pipefail

ISL=${ISL:-8192}
OSL=${OSL:-1024}
CONCUR=${CONCUR:-16}

export MODEL=${MODEL:-/weights/DeepSeek-R1-0528}

export TP_SIZE=${TP_SIZE:-8}   # one host; 128 MLA heads / 8 = 16 per rank
export EP_SIZE=${EP_SIZE:-8}   # experts follow tp
export NNODES=1

export QUANTIZATION=${QUANTIZATION:-fp8}            # blockwise-fp8 checkpoint
export ATTENTION_BACKEND=pallas_tpu_mla             # required: the default rpa_v3 is not MLA
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE-fp8_e4m3}    # 2x the pool of bf16, which is what makes CONCUR=16 fit
export PAGE_SIZE=${PAGE_SIZE:-128}                  # the only page size with tuned MLA blocks at 16 heads
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.90}
export SGLANG_TPU_FP8_MOE_FP32_SCALES=${SGLANG_TPU_FP8_MOE_FP32_SCALES:-1}  # f32 MoE scales, as in the accuracy recipe

export CONTEXT_LEN=${CONTEXT_LEN:-$((ISL + OSL + 64))}
export CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-$ISL}   # a cold ISL-token prompt is one chunk
export MAX_PREFILL_TOKENS=$CHUNKED_PREFILL_SIZE
export MAX_RUNNING=${MAX_RUNNING:-$CONCUR}
export TORCH_COMPILE_MAX_BS=$MAX_RUNNING
# CI compiles cold every run (ENABLE_COMPILE_CACHE=0): one prefill and one decode
# bucket instead of the default power-of-two ladders.
export TPU_COMPILE_TOKENS=${TPU_COMPILE_TOKENS:-$ISL}
export TPU_COMPILE_BS=${TPU_COMPILE_BS:-$CONCUR}

export DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-1}   # random-prompt throughput: a cache hit would skip a real prefill
export ENABLE_PROFILER_ENDPOINT=${ENABLE_PROFILER_ENDPOINT:-0}
# drop mapped checkpoint pages as they load; keeps the CI pod off node-memory eviction
export EXTRA_ARGS="--weight-loader-drop-cache-after-load${EXTRA_ARGS:+ $EXTRA_ARGS}"

cd "$(dirname "$0")/.."
exec ./launch_server.sh "$@"
