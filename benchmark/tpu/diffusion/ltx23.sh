#!/usr/bin/env bash
# LTX-2.3 text/image-to-video + audio (Lightricks/LTX-2.3, materialized through its model
# overlay) on one tpu7x 2x2x1 host (8 cores): Ulysses 8 (32 heads). Both guidance branches
# run in one sequence-parallel group.
set -euo pipefail
export MODEL="${MODEL:-Lightricks/LTX-2.3}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree ${NUM_GPUS}}"
# The 22B DiT stays resident and whole on every core.
export MODEL_ARGS="${MODEL_ARGS---dit-cpu-offload false --use-fsdp-inference false}"
exec bash "$(dirname "$0")/launch.sh" "$@"
