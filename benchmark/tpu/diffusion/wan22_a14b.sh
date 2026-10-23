#!/usr/bin/env bash
# Wan2.2-T2V-A14B / I2V-A14B on one tpu7x 2x2x1 host (8 cores): CFG 2 x Ulysses 2 x ring 2,
# both experts resident.
#   ./wan22_a14b.sh                                          # T2V
#   MODEL=Wan-AI/Wan2.2-I2V-A14B-Diffusers ./wan22_a14b.sh   # I2V
set -euo pipefail
export MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---cfg-parallel-size 2 --ulysses-degree 2 --ring-degree 2}"
export MODEL_ARGS="${MODEL_ARGS---regional-compile}"
exec bash "$(dirname "$0")/launch.sh" "$@"
