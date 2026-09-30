#!/usr/bin/env bash
# Kimi-K2.6 (1.1T MoE, native INT4 compressed-tensors) THROUGHPUT recipe --
# ONE tpu7x host, 4 chips x 2 cores = tp8.
#
# Sample usage:
#   ISL=8192 OSL=1024 CONCUR=16 ./kimi_k26_perf_tp8.sh
#
# NOT YET VALIDATED AT tp8. The int4 expert weights measured 40.9 GB per rank at
# tp16, so ~82 GB of a rank's 101.7 GB here, and the KV pool plus the 8192-token
# extend graph share what is left. If bring-up dies in the precompile, that is
# the first thing to look at: lower CONCUR (the pool) before touching
# MEM_FRACTION_STATIC (the weights still have to fit).
set -uo pipefail

ISL=${ISL:-8192}
OSL=${OSL:-1024}
CONCUR=${CONCUR:-16}

export MODEL=${MODEL:-/weights/Kimi-K2.6}

export TP_SIZE=${TP_SIZE:-8}
# ep=tp. The tp16 recipe's warning against EP=8 does not apply here: it is about
# tp-within-expert=2, whose per-rank intermediate has no 128-aligned divisor for
# the gmm tile. At tp8/ep8 that factor is 1.
export EP_SIZE=${EP_SIZE:-8}
export NNODES=1

# Native int4 (compressed-tensors pack-quantized). "unquant" asks for no online
# quantization; the checkpoint's own quant_method still wins. QUANTIZATION=fp8
# would silently mis-load the int4 experts.
export QUANTIZATION=unquant
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-pallas_tpu_mla}   # Kimi is an MLA model
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE-fp8_e4m3}
export PAGE_SIZE=${PAGE_SIZE:-128}                              # 128 also avoids the long-context page-table SMEM cliff
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.93}

export CONTEXT_LEN=${CONTEXT_LEN:-$((ISL + OSL + 64))}
export CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-$ISL}
export MAX_PREFILL_TOKENS=$CHUNKED_PREFILL_SIZE
export MAX_RUNNING=${MAX_RUNNING:-$CONCUR}
export TORCH_COMPILE_MAX_BS=$MAX_RUNNING
export TPU_COMPILE_TOKENS=${TPU_COMPILE_TOKENS:-$ISL}   # text-only sweep: no vision prefill shapes to cover
export TPU_COMPILE_BS=${TPU_COMPILE_BS:-$CONCUR}

export DISABLE_RADIX_CACHE=${DISABLE_RADIX_CACHE:-1}
export ENABLE_PROFILER_ENDPOINT=${ENABLE_PROFILER_ENDPOINT:-0}
# Custom config/tokenizer/processor code ships in the checkpoint.
export EXTRA_ARGS="--trust-remote-code --weight-loader-drop-cache-after-load${EXTRA_ARGS:+ $EXTRA_ARGS}"
# The VL warmup exercises the multimodal embed-merge path; this sweep is text-only.
: "${SKIP_WARMUP:=1}"
[[ "$SKIP_WARMUP" == "1" ]] && export EXTRA_ARGS="--skip-server-warmup${EXTRA_ARGS:+ $EXTRA_ARGS}"

cd "$(dirname "$0")/.."
exec ./launch_server.sh "$@"
