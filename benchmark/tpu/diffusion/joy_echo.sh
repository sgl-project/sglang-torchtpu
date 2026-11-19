#!/usr/bin/env bash
# JoyAI-Echo text-to-video + audio (jdopensource/JoyAI-Echo, materialized
# through its model overlay). The distilled DiT runs unguided (no CFG branch),
# so all 8 cores take Ulysses (32 heads).
#   ./joy_echo.sh
set -euo pipefail
export MODEL="${MODEL:-jdopensource/JoyAI-Echo}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree ${NUM_GPUS}}"
# The DiT stays resident and whole on every core.
export MODEL_ARGS="${MODEL_ARGS---dit-cpu-offload false --use-fsdp-inference false}"
exec bash "$(dirname "$0")/launch.sh" "$@"
