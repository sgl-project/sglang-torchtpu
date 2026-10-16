#!/usr/bin/env bash
# GLM-5.3-Flash text serving (TP8, DSA). The server settings live in the Python
# config recipe/configs/glm_5_3_flash/server.py; this wrapper is for launchers
# that run shell recipes.
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec "${PYTHON:-python3}" -m benchmark.tpu.launch_server \
  --config recipe/configs/glm_5_3_flash/server.py "$@"
