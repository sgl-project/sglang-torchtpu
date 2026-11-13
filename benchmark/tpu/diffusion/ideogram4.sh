#!/usr/bin/env bash
# Ideogram 4 text-to-image (ideogram-ai/ideogram-4-fp8). The conditional and
# unconditional DiTs run in turn, each sharding its packed rows over all 8
# cores (18 heads, padded to 24).
#   ./ideogram4.sh
set -euo pipefail
export MODEL="${MODEL:-ideogram-ai/ideogram-4-fp8}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree ${NUM_GPUS}}"
# Both DiTs stay resident and whole on every core.
export MODEL_ARGS="${MODEL_ARGS---dit-cpu-offload false --use-fsdp-inference false}"
exec bash "$(dirname "$0")/launch.sh" "$@"
