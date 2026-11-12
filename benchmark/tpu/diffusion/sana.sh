#!/usr/bin/env bash
# SANA 1.5 text-to-image (Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers).
# The DiT has no sequence parallelism; one core runs both guidance branches.
#   ./sana.sh
set -euo pipefail
export MODEL="${MODEL:-Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers}"
export NUM_GPUS="${NUM_GPUS:-1}"
export PARALLEL_ARGS="${PARALLEL_ARGS-}"
export MODEL_ARGS="${MODEL_ARGS---dit-cpu-offload false}"
exec bash "$(dirname "$0")/launch.sh" "$@"
