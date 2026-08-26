#!/usr/bin/env bash
# DeepSeek-R1-0528 (671B MoE, block-fp8) multi-host launch recipe -- 2 hosts, tp=16.
#   SGLANG_USE_RAY=1 NNODES=2 ./deepseek_r1_0528_mh.sh
set -uo pipefail

export MODEL=${MODEL:-/weights/DeepSeek-R1-0528}

# --- parallelism ---
export NNODES=${NNODES:-2}
export TP_SIZE=${TP_SIZE:-16}
export EP_SIZE=${EP_SIZE:-16}

# --- attention / precision ---
export ATTENTION_BACKEND=pallas_tpu_mla   # required; the default rpa_v3 is not MLA
export KV_CACHE_DTYPE=""                  # required; pallas_tpu_mla raises on fp8 KV
export QUANTIZATION=${QUANTIZATION:-fp8}

# --- memory / shapes ---
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.70}
export MAX_RUNNING=${MAX_RUNNING:-128}
export PAGE_SIZE=${PAGE_SIZE:-64}
export CONTEXT_LEN=${CONTEXT_LEN:-40960}  # 32768 gen + 8192 prompt headroom
export CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
export MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-8192}
export TORCH_COMPILE_MAX_BS=${TORCH_COMPILE_MAX_BS:-$MAX_RUNNING}

# drop mapped checkpoint pages as they load; keeps the CI pod off node-memory eviction
export EXTRA_ARGS="--reasoning-parser deepseek-r1 --weight-loader-drop-cache-after-load${EXTRA_ARGS:+ $EXTRA_ARGS}"
export ENABLE_PROFILER_ENDPOINT=${ENABLE_PROFILER_ENDPOINT:-0}

cd "$(dirname "$0")/.."
exec ./launch_server.sh "$@"
