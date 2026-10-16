#!/usr/bin/env bash
# Shell entry for recipe/configs/glm_5_3_flash/server.py (for shell launchers).
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec "${PYTHON:-python3}" -m benchmark.tpu.launch_server \
  --config recipe/configs/glm_5_3_flash/server.py "$@"
