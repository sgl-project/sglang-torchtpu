#!/usr/bin/env bash
set -euo pipefail
cd /workspace/sglang-torchtpu
exec python3 -u -m benchmark.tpu.glm52_baseline.allocation
