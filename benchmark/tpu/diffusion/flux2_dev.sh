#!/usr/bin/env bash
# FLUX.2-dev text-to-image on one tpu7x 2x2x1 host (8 cores): Ulysses 8 (48 heads), DiT resident,
# Mistral3 text encoder folded across the cores.
#   ./flux2_dev.sh
set -euo pipefail
export MODEL="${MODEL:-black-forest-labs/FLUX.2-dev}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree ${NUM_GPUS}}"
export MODEL_ARGS="${MODEL_ARGS---dit-cpu-offload false --use-fsdp-inference false}"
exec bash "$(dirname "$0")/launch.sh" "$@"
