#!/usr/bin/env bash
# Cosmos3-Super text/image-to-video (nvidia/Cosmos3-Super, 64B). The 128 GB of
# weights shard over the 8 cores (FSDP inference); the visual tokens shard by
# Ulysses (64 query / 8 key-value heads).
#   ./cosmos3_super.sh
set -euo pipefail
export MODEL="${MODEL:-nvidia/Cosmos3-Super}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree ${NUM_GPUS}}"
export MODEL_ARGS="${MODEL_ARGS---use-fsdp-inference true --dit-cpu-offload false}"
exec bash "$(dirname "$0")/launch.sh" "$@"
