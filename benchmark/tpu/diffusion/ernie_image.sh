#!/usr/bin/env bash
# ERNIE-Image text-to-image.
# 32 heads: the Ulysses degree must divide 32.
#   ./ernie_image.sh                      # 8 cores, Ulysses 8 with serial CFG
# The prompt enhancer runs per request unless it sends "use_pe": false.
set -euo pipefail
export MODEL="${MODEL:-baidu/ERNIE-Image}"
export NUM_GPUS="${NUM_GPUS:-8}"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree ${NUM_GPUS}}"
fi
# The DiT stays resident and whole on every core; each block compiles as one
# program.
export MODEL_ARGS="${MODEL_ARGS---dit-cpu-offload false --use-fsdp-inference false --regional-compile}"
exec bash "$(dirname "$0")/launch.sh" "$@"
