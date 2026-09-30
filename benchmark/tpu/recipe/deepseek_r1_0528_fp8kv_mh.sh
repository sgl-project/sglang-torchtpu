#!/usr/bin/env bash
# DeepSeek-R1-0528 accuracy server, tp16/ep16 fp8 KV on one 2x2x2 slice.
# The JobSet runs this on every host of the slice.
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec python3 -m benchmark.tpu.launch_server --config recipe/configs/deepseek_r1_0528/accuracy.py \
    --working-directory benchmark/tpu "$@"
