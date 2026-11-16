#!/usr/bin/env bash
# SANA-Video 2B 720p text-to-video on one tpu7x chip: CFG 2 (one branch per core), the whole
# DiT resident and compiled as one program.
#   ./sana_video.sh
#   NUM_GPUS=1 ./sana_video.sh                 # one core
#   ./sana_video.sh --vae-precision fp32       # fp32 VAE (default bf16)
set -euo pipefail
export MODEL="${MODEL:-Efficient-Large-Model/SANA-Video_2B_720p_diffusers}"
export NUM_GPUS="${NUM_GPUS:-2}"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  export PARALLEL_ARGS="${PARALLEL_ARGS---cfg-parallel-size 2}"
fi
# Not redundant: without it the video defaults CPU-offload the DiT. Server warmup
# compiles the served shape (its defaults cap video at 832x480, 17 frames).
export MODEL_ARGS="${MODEL_ARGS---dit-layerwise-offload false --warmup-resolutions 1280x704 --warmup-num-frames 81}"
exec bash "$(dirname "$0")/launch.sh" "$@"
