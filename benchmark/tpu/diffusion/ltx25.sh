#!/usr/bin/env bash
# LTX-2.5 text/image-to-video + audio. The distilled DiT runs unguided (no CFG
# branch), so all 8 cores take Ulysses (32 heads).
#   ./ltx25.sh
set -euo pipefail
export MODEL="${MODEL:-Lightricks/LTX-2.5-Diffusers}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree ${NUM_GPUS}}"
# The 22B DiT stays resident and whole on every core.
export MODEL_ARGS="${MODEL_ARGS---dit-cpu-offload false --use-fsdp-inference false}"
exec bash "$(dirname "$0")/launch.sh" "$@"
