#!/usr/bin/env bash
# Wan2.2-Animate-2-14B on one tpu7x 2x2x1 host (8 cores): Ulysses 8, the base checkpoint's CFG branches serial,
# DiT and encoders resident.
#   ./wan_animate_2.sh                                                          # base
#   MODEL=Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers ./wan_animate_2.sh    # distilled
set -euo pipefail
export MODEL="${MODEL:-Wan-AI/Wan2.2-Animate-2-14B-Diffusers}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree 8}"
export MODEL_ARGS="${MODEL_ARGS---regional-compile --dit-layerwise-offload false --image-encoder-cpu-offload false}"
exec bash "$(dirname "$0")/launch.sh" "$@"
