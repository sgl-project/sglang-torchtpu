#!/usr/bin/env bash
# Anima text-to-image (guided): CFG 2 x Ulysses 4 on 8 cores (16 heads).
#   ./anima.sh
set -euo pipefail
export MODEL="${MODEL:-circlestone-labs/Anima-Base-v1.0-Diffusers}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---cfg-parallel-size 2 --ulysses-degree 4}"
# The 2B DiT stays resident and whole on every core.
export MODEL_ARGS="${MODEL_ARGS---dit-cpu-offload false --use-fsdp-inference false}"
exec bash "$(dirname "$0")/launch.sh" "$@"
