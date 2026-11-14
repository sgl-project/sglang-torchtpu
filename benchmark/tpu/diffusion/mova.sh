#!/usr/bin/env bash
# MOVA video + audio (OpenMOSS-Team/MOVA-720p): Ulysses 8 with serial CFG (40 video
# heads; the 12 audio heads pad to 16), compiling each DiT block.
#   ./mova.sh
set -euo pipefail
export MODEL="${MODEL:-OpenMOSS-Team/MOVA-720p}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree 8}"
export MODEL_ARGS="${MODEL_ARGS---regional-compile --dit-cpu-offload false --use-fsdp-inference false}"
exec bash "$(dirname "$0")/launch.sh" "$@"
