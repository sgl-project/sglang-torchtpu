#!/usr/bin/env bash
# LingBot-World realtime world model on one tpu7x 2x2x1 host (8 cores): Ulysses 8 over the 40 heads,
# serial CFG (DMD, guidance 1.0). Sessions stream over ws://HOST:PORT/v1/realtime_video/generate:
# 3-latent-frame chunks (9 frames, then 12 a chunk) at 832x480, 4 DMD steps each, over a 9-frame
# sink and an 18-frame K/V window, steered by camera actions (w/a/s/d move, i/j/k/l turn).
#   ./lingbot_world.sh                                                 # v2 14B
#   MODEL=robbyant/lingbot-world-fast-diffusers ./lingbot_world.sh     # v1
set -euo pipefail
export MODEL="${MODEL:-robbyant/lingbot-world-v2-14b-causal-fast-diffusers}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---cfg-parallel-size 1 --ulysses-degree ${NUM_GPUS}}"
# --regional-compile compiles the 40 blocks; the KV-cache attention runs eagerly between their
# halves. --dit-cpu-offload false keeps the DiT resident on each core. Realtime serving has no
# server warmup: the first session compiles. Warm up with an 8-chunk session whose camera script
# ends mid-session, as the blocks compile for chunks with and without a camera condition; it
# covers the six K/V lengths and the cache roll too.
export MODEL_ARGS="${MODEL_ARGS---pipeline-class-name LingBotWorldCausalDMDPipeline --regional-compile --dit-cpu-offload false}"
exec bash "$(dirname "$0")/launch.sh" "$@"
