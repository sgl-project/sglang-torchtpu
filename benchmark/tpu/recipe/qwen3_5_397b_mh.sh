#!/usr/bin/env bash
# Qwen3.5-397B-A17B VL server, tp16/ep16 bf16 on one 2x2x2 slice.
# The JobSet runs this on every host of the slice.
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec python3 -m benchmark.tpu.launch_server --config recipe/configs/qwen3_5_397b/server.py \
    --working-directory benchmark/tpu "$@"
