#!/usr/bin/env bash
# Qwen-Image text-to-image (MODEL=Qwen/Qwen-Image-Edit-2511 for editing).
# 24 heads: Ulysses degree must divide 24.
#   ./qwen_image.sh                       # 8 cores, Ulysses 8
#   NUM_GPUS=1 ./qwen_image.sh
set -euo pipefail
export MODEL="${MODEL:-Qwen/Qwen-Image}"
export NUM_GPUS="${NUM_GPUS:-8}"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree ${NUM_GPUS}}"
fi
# The DiT stays resident and whole on every core.
export MODEL_ARGS="${MODEL_ARGS---dit-cpu-offload false --use-fsdp-inference false}"
exec bash "$(dirname "$0")/launch.sh" "$@"
