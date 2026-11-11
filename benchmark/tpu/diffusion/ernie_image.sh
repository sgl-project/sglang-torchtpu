#!/usr/bin/env bash
# ERNIE-Image text-to-image.
# 32 heads: the Ulysses degree must divide 32.
#   ./ernie_image.sh                      # 8 cores, CFG 2 x Ulysses 4
# The prompt enhancer runs per request unless it sends "use_pe": false, or
# --pe-server-url serves it from an SGLang server.
set -euo pipefail
export MODEL="${MODEL:-baidu/ERNIE-Image}"
export NUM_GPUS="${NUM_GPUS:-8}"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  export PARALLEL_ARGS="${PARALLEL_ARGS---cfg-parallel-size 2 --ulysses-degree $((NUM_GPUS / 2))}"
fi
# The DiT stays resident and whole on every core.
export MODEL_ARGS="${MODEL_ARGS---dit-cpu-offload false --use-fsdp-inference false}"
exec bash "$(dirname "$0")/launch.sh" "$@"
