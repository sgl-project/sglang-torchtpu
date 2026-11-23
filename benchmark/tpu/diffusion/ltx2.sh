#!/usr/bin/env bash
# LTX-2 text-to-video + audio on one tpu7x 2x2x1 host (8 cores): Ulysses 8 (32 heads).
# Both guidance branches run in one sequence-parallel group, so identical requests give
# identical videos.
set -euo pipefail
export MODEL="${MODEL:-Lightricks/LTX-2}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree ${NUM_GPUS}}"
exec bash "$(dirname "$0")/launch.sh" "$@"
