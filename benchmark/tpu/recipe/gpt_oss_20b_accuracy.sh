#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec python3 -m benchmark.tpu.recipe.configs.accuracy --config gpt_oss_20b_accuracy "$@"
