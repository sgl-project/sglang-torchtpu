#!/usr/bin/env bash
# Ming-Image text-to-image.
#   ./ming_image.sh                       # 8 cores, Ulysses 8 (30 heads, padded to 32)
set -euo pipefail
export MODEL="${MODEL:-inclusionAI/Ming-Image-0.1-Design}"
export NUM_GPUS="${NUM_GPUS:-8}"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree ${NUM_GPUS}}"
fi
# Read checkpoints on the host, one tensor at a time: the streamer stages whole
# files in device memory, and the encoder's 5 GB shards then exhaust a core.
export SGLANG_USE_RUNAI_MODEL_STREAMER="${SGLANG_USE_RUNAI_MODEL_STREAMER:-false}"
# The DiT stays resident and whole on every core.
export MODEL_ARGS="${MODEL_ARGS---dit-cpu-offload false --use-fsdp-inference false}"
# The 17B MoE encoder is replicated on every core: tensor parallelism would
# split each expert's 512 columns across the cores.
# Per-layer compile with spmd-safe regions: 2.77 s per 2048x2048 image against
# 2.82 s for the whole DiT, which also compiles 4x longer.
exec bash "$(dirname "$0")/launch.sh" --encoder-parallel replicate --regional-compile "$@"
