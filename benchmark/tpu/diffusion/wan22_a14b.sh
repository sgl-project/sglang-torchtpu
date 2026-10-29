#!/usr/bin/env bash
# Wan2.2-T2V-A14B / I2V-A14B on one tpu7x 2x2x1 host (8 cores): CFG 2 x Ulysses 2 x ring 2,
# both experts resident.
#   ./wan22_a14b.sh                                          # T2V
#   MODEL=Wan-AI/Wan2.2-I2V-A14B-Diffusers ./wan22_a14b.sh   # I2V
set -euo pipefail
export MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---cfg-parallel-size 2 --ulysses-degree 2 --ring-degree 2}"
# Not redundant: without it the video defaults CPU-offload the DiT, and every request copies
# the experts in from the host (T2V 1280x720x81: 125 s instead of 58 s).
export MODEL_ARGS="${MODEL_ARGS---regional-compile --dit-layerwise-offload false}"
exec bash "$(dirname "$0")/launch.sh" "$@"
