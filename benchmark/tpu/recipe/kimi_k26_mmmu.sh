#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec python3 -m benchmark.tpu.recipe.configs.accuracy --config kimi_k26_mmmu "$@"
