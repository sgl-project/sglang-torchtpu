#!/usr/bin/env bash
# Kimi-K2.6 MMMU-Pro server, tp16/ep16 on one 2x2x2 slice.
# The JobSet runs this on every host of the slice.
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec python3 -m benchmark.tpu.launch_server --config recipe/configs/kimi_k26/mmmu.py \
    --working-directory benchmark/tpu "$@"
