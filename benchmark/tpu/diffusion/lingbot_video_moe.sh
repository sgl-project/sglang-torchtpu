#!/usr/bin/env bash
# LingBot-Video-MoE 30B-A3B text-to-video on one tpu7x 2x2x1 host (8 cores): Ulysses 8 with
# serial CFG, the DiT resident and whole on every core.
set -euo pipefail
export MODEL="${MODEL:-robbyant/lingbot-video-moe-30b-a3b}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree 8}"
# Read checkpoints on the host, one tensor at a time: the streamer stages whole
# files in device memory next to the 60 GB DiT.
export SGLANG_USE_RUNAI_MODEL_STREAMER="${SGLANG_USE_RUNAI_MODEL_STREAMER:-false}"
# The TPU DiT forward compiles per block. Without the offload flags the video
# defaults CPU-offload the DiT and copy it in from the host for every request.
# The server warmup compiles the default 81 frames (it caps them at 17 otherwise).
export MODEL_ARGS="${MODEL_ARGS---regional-compile --use-fsdp-inference false --dit-cpu-offload false --dit-layerwise-offload false --warmup-num-frames 81}"
exec bash "$(dirname "$0")/launch.sh" "$@"
