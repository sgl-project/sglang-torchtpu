#!/usr/bin/env bash
# LongCat-Image text-to-image (meituan-longcat/LongCat-Image). Ulysses 8 (24
# heads); the Qwen2.5-VL text encoder, which also rewrites the prompt, is
# replicated on every core.
#   ./longcat_image.sh
set -euo pipefail
export MODEL="${MODEL:-meituan-longcat/LongCat-Image}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree ${NUM_GPUS}}"
export MODEL_ARGS="${MODEL_ARGS---dit-cpu-offload false --use-fsdp-inference false}"
exec bash "$(dirname "$0")/launch.sh" --encoder-parallel replicate "$@"
