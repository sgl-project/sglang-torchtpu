#!/usr/bin/env bash
# Qwen-Image 2.1 text-to-image and editing. 32 heads: Ulysses degree up to 8.
#   ./qwen_image21.sh                    # 8 cores, Ulysses 8
#   NUM_GPUS=1 ./qwen_image21.sh
set -euo pipefail
export MODEL="${MODEL:-Qwen/Qwen-Image-2.1}"
export NUM_GPUS="${NUM_GPUS:-8}"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree ${NUM_GPUS}}"
fi
# The 7B DiT stays resident and whole on every core.
export MODEL_ARGS="${MODEL_ARGS---dit-cpu-offload false --use-fsdp-inference false}"
exec bash "$(dirname "$0")/launch.sh" "$@"
