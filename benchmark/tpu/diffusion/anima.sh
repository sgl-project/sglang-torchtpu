#!/usr/bin/env bash
# Anima text-to-image (guided): Ulysses over the cores with serial CFG (16 heads).
#   ./anima.sh                 # 8 cores, Ulysses 8
#   NUM_GPUS=2 ./anima.sh      # one chip, Ulysses 2
set -euo pipefail
export MODEL="${MODEL:-circlestone-labs/Anima-Base-v1.0-Diffusers}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree ${NUM_GPUS}}"
# The 2B DiT stays resident and whole on every core.
export MODEL_ARGS="${MODEL_ARGS---dit-cpu-offload false --use-fsdp-inference false}"
exec bash "$(dirname "$0")/launch.sh" "$@"
