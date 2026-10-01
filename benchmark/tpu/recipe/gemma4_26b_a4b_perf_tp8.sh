#!/usr/bin/env bash
# Gemma-4-26B-A4B-it (MoE) throughput server, tp8/ep8 on one tpu7x host. The
# Python config owns the settings; this is the shell entry point the perf
# harness launches.
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec python3 -m benchmark.tpu.launch_server --config recipe/configs/gemma4_26b_a4b/perf.py \
    --working-directory benchmark/tpu "$@"
