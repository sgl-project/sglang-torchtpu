#!/usr/bin/env bash
# VDN-H3 text- and first/last-frame-to-video with audio, 8 NFE, on one tpu7x 2x2x1 host (8 cores):
# Ulysses 8 over a replicated DiT, the hybrid window attention compiled per DiT block. The text
# encoder stays on the host, the VAEs on the TPU; 4 s clips also fit --text-encoder-cpu-offload false.
# The first launch materializes the model overlay (~76 GB) under SGLANG_DIFFUSION_CACHE_ROOT.
set -euo pipefail
export MODEL="${MODEL:-OpenVDN/vdn-minimax-h3}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree 8}"
export MODEL_ARGS="${MODEL_ARGS---use-fsdp-inference false --dit-cpu-offload false --regional-compile --performance-mode manual --warmup-mode off}"
exec bash "$(dirname "$0")/launch.sh" --attention-backend hybrid_window_attn_h3 \
  --text-encoder-cpu-offload true --vae-cpu-offload false "$@"
