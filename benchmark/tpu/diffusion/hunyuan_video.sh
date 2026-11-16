#!/usr/bin/env bash
# HunyuanVideo text-to-video (embedded guidance, no CFG branch) on one tpu7x 2x2x1 host.
#   ./hunyuan_video.sh                    # 8 cores, Ulysses 8 (24 heads)
set -euo pipefail
export MODEL="${MODEL:-hunyuanvideo-community/HunyuanVideo}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree ${NUM_GPUS}}"
# The 13B DiT stays resident and whole on every core. The default video warmup
# runs 720x720 at 29 frames, a shape no request uses: warm up the served one.
export MODEL_ARGS="${MODEL_ARGS---dit-cpu-offload false --use-fsdp-inference false --warmup-resolutions 1280x720 --warmup-num-frames 125}"
# Per-block compile.
exec bash "$(dirname "$0")/launch.sh" --regional-compile "$@"
