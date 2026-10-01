#!/usr/bin/env bash
# Gemma-4-31B-it throughput server, tp8 on one tpu7x host. The Python config
# owns the settings; this is the shell entry point the perf harness launches.
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec python3 -m benchmark.tpu.launch_server --config recipe/configs/gemma4_31b/perf.py \
    --working-directory benchmark/tpu "$@"
