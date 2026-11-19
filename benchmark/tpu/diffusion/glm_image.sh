#!/usr/bin/env bash
# GLM-Image text-to-image (zai-org/GLM-Image): the autoregressive prior
# tokens decode on the main rank; the DiT runs at Ulysses 8 (32 heads).
#   ./glm_image.sh
set -euo pipefail
export MODEL="${MODEL:-zai-org/GLM-Image}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree ${NUM_GPUS}}"
export MODEL_ARGS="${MODEL_ARGS---dit-cpu-offload false --use-fsdp-inference false}"
exec bash "$(dirname "$0")/launch.sh" "$@"
