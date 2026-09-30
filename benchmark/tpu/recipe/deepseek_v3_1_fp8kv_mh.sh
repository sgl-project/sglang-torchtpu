#!/usr/bin/env bash
# DeepSeek-V3.1 accuracy server, tp16/ep16 fp8 KV on one 2x2x2 slice.
# The JobSet runs this on every host of the slice.
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec python3 -m benchmark.tpu.launch_server --config recipe/configs/deepseek_v3_1/accuracy.py \
    --working-directory benchmark/tpu "$@"
