#!/usr/bin/env bash
# Qwen3-VL-30B-A3B-Instruct (bf16 MoE decoder + ViT) on one tpu7x host — 4 chips x 2 cores
# = tp8, with the 128 experts split ep8 (16 per rank).
#
# recipe/configs/qwen3_vl_30b_a3b/pd_singlehost.py serves this same
# configuration, adding only the disaggregated handoff.
#
# Sample usage:
#   ./server.sh                                                  # 8k/1k serving shape
#   MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct ./server.sh             # weights from HF
set -euo pipefail

# /weights is the CI host's mount; pass MODEL for a Hugging Face id or a local copy.
export MODEL=${MODEL:-/weights/Qwen3-VL-30B-A3B-Instruct}
export TP_SIZE=8
# Experts expert-parallel rather than tensor-parallel, like the text 30B-A3B
# recipes: moe_intermediate_size 768 / tp8 = 96 would not fill a 128-lane tile.
export EP_SIZE=${EP_SIZE:-8}

# bf16 checkpoint. "unquant" is the explicit opt-out, not merely an unset: it
# also sets _quantization_explicitly_unset, so the model overrides cannot
# re-detect a quant method. Empty would only drop launch_server.sh's fp8.
export QUANTIZATION=unquant
# No KV scales in the checkpoint: fp8_e4m3 would scale through factors of 1.0.
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-bfloat16}

export PAGE_SIZE=${PAGE_SIZE:-128}
export CONTEXT_LEN=${CONTEXT_LEN:-9216}
export CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
export MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-8192}
export MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-262144}

# reconcile_tpu_batch_limits raises torch_compile_max_bs back to max_running_requests,
# so the two move together. 8 gives bs buckets [1, 2, 4, 8].
export CONC=${CONC:-8}
export TORCH_COMPILE_MAX_BS=${TORCH_COMPILE_MAX_BS:-8}
# The VL EXTEND grid multiplies these buckets by its input variants; four bound
# the startup. 8192 must stay to cover MAX_PREFILL_TOKENS.
export TPU_COMPILE_TOKENS=${TPU_COMPILE_TOKENS:-"512 1024 2048 8192"}
# ~61 GB of bf16 weights is ~7.6 GB per rank, so the budget is dominated by
# the KV pool and graph temporaries: the ViT and the decoder each capture their own.
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.30}

# max_pixels caps one image at ~306 vision tokens (patch 16 + 2x2 merge = 1024
# px per token). Uncapped, a single MMMU-Pro page exceeds the whole 8k chunk.
MM_PROCESS_CONFIG='{"image":{"min_pixels":65536,"max_pixels":313600}}'
# --watchdog-timeout: the EXTEND/decode graph captures outlast the 300s default
# and trip the watchdog into repeated stack dumps mid-compile.
# direct tracing compiles the ViT buckets and the VL decoder graphs
# concurrently (#1216), and --use-spmd-compile compiles each SPMD-safe warmup
# graph on one owner rank and broadcasts the executable across TP ranks
# (#1623: cold start 303 s -> 117 s, peak host RSS 422 -> 189 GiB, same
# MMMU-Pro score, tpu7x 2x2x1, cache off). A caller's EXTRA_ARGS follows and
# can pass --no-use-spmd-compile or --tpu-compile-tracer dynamo to compare.
export EXTRA_ARGS="--trust-remote-code --watchdog-timeout 3600 \
--tpu-compile-tracer direct --use-spmd-compile \
--served-model-name ${SERVED_MODEL_NAME:-qwen3-vl-30b-a3b} \
--mm-process-config $MM_PROCESS_CONFIG${EXTRA_ARGS:+ $EXTRA_ARGS}"

cd "$(dirname "$0")/../.."
exec ./launch_server.sh "$@"
