#!/usr/bin/env bash
# LongLive-2.0-5B text/image-to-video on one tpu7x core: 8-latent-frame causal chunks over an
# 8-frame sink and a 32-frame KV window, 4 steps, no CFG (guidance 1.0).
set -euo pipefail
export MODEL="${MODEL:-Rabinovich/LongLive-2.0-5B-Diffusers}"
export NUM_GPUS="${NUM_GPUS:-1}"
export PARALLEL_ARGS="${PARALLEL_ARGS-}"
# --dit-cpu-offload false keeps the DiT resident on the core. --warmup-mode request runs the
# first request once as warmup at its own size: the server warmup of this TI2V model conditions
# on a 64x64 image, which sets 960x928.
export MODEL_ARGS="${MODEL_ARGS---regional-compile --dit-cpu-offload false --warmup-mode request}"
# The VAE decoder calls after the second, which run with full causal caches, compile to one graph
# per frame size; the first two calls stay eager.
export SGLANG_DIFFUSION_ENABLE_WAN_VAE_DECODER_COMPILE="${SGLANG_DIFFUSION_ENABLE_WAN_VAE_DECODER_COMPILE:-1}"
exec bash "$(dirname "$0")/launch.sh" "$@"
