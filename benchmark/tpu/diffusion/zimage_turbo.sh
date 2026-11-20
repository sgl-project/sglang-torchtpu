#!/usr/bin/env bash
# Z-Image-Turbo text-to-image. Ulysses spans the cores (30 heads, padded to the degree), with
# per-layer compiled regions that keep each layer's collectives in one program.
#   ./zimage_turbo.sh                   # 8 cores, Ulysses 8
#   NUM_GPUS=1 ./zimage_turbo.sh
set -euo pipefail
export MODEL="${MODEL:-Tongyi-MAI/Z-Image-Turbo}"
export NUM_GPUS="${NUM_GPUS:-8}"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree ${NUM_GPUS}}"
  export MODEL_ARGS="${MODEL_ARGS---regional-compile}"
fi
exec bash "$(dirname "$0")/launch.sh" "$@"
