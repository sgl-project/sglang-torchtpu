#!/usr/bin/env bash
# Z-Image text-to-image (CFG with a negative prompt, 50 steps). Ulysses spans the cores
# (30 heads, padded to the degree), with per-layer compiled regions that keep each
# layer's collectives in one program.
#   ./zimage.sh                   # 8 cores, Ulysses 8
#   NUM_GPUS=1 ./zimage.sh
set -euo pipefail
export MODEL="${MODEL:-Tongyi-MAI/Z-Image}"
export NUM_GPUS="${NUM_GPUS:-8}"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree ${NUM_GPUS}}"
  export MODEL_ARGS="${MODEL_ARGS---regional-compile}"
fi
exec bash "$(dirname "$0")/launch.sh" "$@"
