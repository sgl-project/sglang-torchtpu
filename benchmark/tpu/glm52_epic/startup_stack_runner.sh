#!/usr/bin/env bash
# Diagnostic only: inherited by spawned Python workers, never performance runs.
set -euo pipefail
export GLM52_STARTUP_STACK_DIR=/tmp/tpu_logs/glm52-startup-stacks
export GLM52_STARTUP_STACK_SECONDS=300
export PYTHONPATH="$PWD/benchmark/tpu/glm52_epic/observer${PYTHONPATH:+:$PYTHONPATH}"
python3 benchmark/tpu/glm52_epic/startup_memory.py &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
bash benchmark/tpu/glm52_epic/torch_8chip_runner.sh
