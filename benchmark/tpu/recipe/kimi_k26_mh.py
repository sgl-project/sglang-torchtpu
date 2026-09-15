#!/usr/bin/env bash
# Kimi-K2.6 (1.1T MoE, native INT4 compressed-tensors) + MoonViT multi-host
# launch recipe -- PARTIALLY VALIDATED.
#
# STATUS: the shapes below were validated on Kimi-K2.5, which is the same
# architecture (config.json architectures == KimiK25ForConditionalGeneration for
# both), the same compressed-tensors pack-quantized int4 format (group_size 32,
# symmetric) and the same MoonViT tower. They have NOT been run against the K2.6
# checkpoint itself, and the nightly consuming this file is registered disabled
# because the model support it needs is not merged yet. Drop the markers once a
# green run exists.
#
# Sizing rationale:
#   1.1T total, int4 experts -> ~37GB/device measured on K2.5 at tp=16 over 2
#   hosts. MEM_FRACTION_STATIC=0.6 left ~45GB for the KV pool at CONTEXT_LEN
#   8192; raising it starves KV, lowering it wastes HBM.
#   ep=16 matches tp. EP=8 leaves tp-within-expert=2, whose per-rank intermediate
#   has no 128-aligned divisor for the gmm tile ("pad the dim" ValueError) -- do
#   not use.
#
# QUANTIZATION= must stay explicitly EMPTY: launch_server.sh defaults to
# ${QUANTIZATION-fp8}, and passing fp8 silently mis-loads the int4 experts.
# Empty lets sglang auto-detect compressed-tensors from the checkpoint config.
#
# TPU_COMPILE_TOKENS must cover the prefill shapes vision requests produce: a
# 448x448 image is 256 vision tokens after the 4:1 merge, and a prefill shape
# outside the precompiled buckets takes the recompile path that crashes
# torch_tpu.
#
# Weights: /weights/Kimi-K2.6 on the shared model-weights-rox ROX Hyperdisk.
set -uo pipefail

export MODEL=${MODEL:-/weights/Kimi-K2.6}

# --- parallelism ---
export NNODES=${NNODES:-2}
export TP_SIZE=${TP_SIZE:-16}
export EP_SIZE=${EP_SIZE:-16}

# --- precision ---
# Native int4 (compressed-tensors pack-quantized). Both MUST be empty; see above.
export QUANTIZATION=
export KV_CACHE_DTYPE=

# --- attention ---
# Kimi is an MLA model.
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-pallas_tpu_mla}

# --- memory / shapes ---
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.6}
export MAX_RUNNING=${MAX_RUNNING:-8}
export CONTEXT_LEN=${CONTEXT_LEN:-8192}
export CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
export MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-8192}
export TPU_COMPILE_TOKENS=${TPU_COMPILE_TOKENS:-"64 128 256 512 1024 2048 4096 8192"}
export TORCH_COMPILE_MAX_BS=${TORCH_COMPILE_MAX_BS:-$MAX_RUNNING}

# Custom config/tokenizer/processor code (configuration_kimi_k25.py,
# kimi_k25_processor.py) ships in the checkpoint.
export EXTRA_ARGS="--trust-remote-code${EXTRA_ARGS:+ $EXTRA_ARGS}"

# sglang's VL warmup exercises the multimodal embed-merge path. That aborted
# XLA:TPU's fusion emitter until main's int32 `masked_scatter_` ATen kernel
# landed (hardware_backend/tpu/patches.py). SKIP_WARMUP=0 once a green run
# confirms the kernel makes it safe.
: "${SKIP_WARMUP:=1}"
[[ "$SKIP_WARMUP" == "1" ]] && export EXTRA_ARGS="--skip-server-warmup${EXTRA_ARGS:+ $EXTRA_ARGS}"

export ENABLE_PROFILER_ENDPOINT=${ENABLE_PROFILER_ENDPOINT:-0}

cd "$(dirname "$0")/.."
exec ./launch_server.sh "$@"
