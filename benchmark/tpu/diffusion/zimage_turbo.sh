#!/usr/bin/env bash
# Z-Image-Turbo text-to-image. 30 heads: power-of-two Ulysses degree is 1 or 2.
#   NUM_GPUS=1 ./zimage_turbo.sh
#   NUM_GPUS=2 ./zimage_turbo.sh          # Ulysses 2
set -euo pipefail
export MODEL="${MODEL:-Tongyi-MAI/Z-Image-Turbo}"
export NUM_GPUS="${NUM_GPUS:-1}"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree ${NUM_GPUS}}"
fi
exec bash "$(dirname "$0")/launch.sh" "$@"
