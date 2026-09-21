#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
exec python3 -m benchmark.tpu.recipe.configs.accuracy --config deepseek_r1_0528_fp8kv_mh "$@"
