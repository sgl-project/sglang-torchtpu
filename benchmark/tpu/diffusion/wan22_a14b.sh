#!/usr/bin/env bash
# Wan2.2-T2V-A14B / I2V-A14B on one tpu7x 2x2x1 host (8 cores), both experts resident.
# T2V runs CFG 2 x ring 4 on the Pallas ring kernel. I2V runs CFG 2 x Ulysses 2 x ring 2: its
# output follows the input image's aspect ratio, and sizes that need tail padding (624x624x81 is
# 31941 tokens) take the K/V all-gather ring, which is cheaper over 2 ranks than over 4 (18.3 s
# vs 23.3 s per request; an unpadded 16:9 1280x720 request is 60.2 s vs 55.9 s at ring 4).
#   ./wan22_a14b.sh                                          # T2V
#   MODEL=Wan-AI/Wan2.2-I2V-A14B-Diffusers ./wan22_a14b.sh   # I2V
set -euo pipefail
export MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
export NUM_GPUS="${NUM_GPUS:-8}"
case "${MODEL}" in
  *I2V*) default_parallel="--cfg-parallel-size 2 --ulysses-degree 2 --ring-degree 2" ;;
  *) default_parallel="--cfg-parallel-size 2 --ulysses-degree 1 --ring-degree 4" ;;
esac
export PARALLEL_ARGS="${PARALLEL_ARGS-${default_parallel}}"
# Not redundant: without it the video defaults CPU-offload the DiT, and every request copies
# the experts in from the host (T2V 1280x720x81: 125 s instead of 58 s).
export MODEL_ARGS="${MODEL_ARGS---regional-compile --dit-layerwise-offload false}"
exec bash "$(dirname "$0")/launch.sh" "$@"
