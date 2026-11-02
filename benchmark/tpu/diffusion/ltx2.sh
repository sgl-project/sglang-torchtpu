#!/usr/bin/env bash
# LTX-2 text-to-video + audio on one tpu7x 2x2x1 host (8 cores): CFG 2 x Ulysses 4.
set -euo pipefail
export MODEL="${MODEL:-Lightricks/LTX-2}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---cfg-parallel-size 2 --ulysses-degree 4}"
exec bash "$(dirname "$0")/launch.sh" "$@"
