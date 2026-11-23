#!/usr/bin/env bash
# HunyuanImage-3.0 (80B MoE, ~13B active): text-to-image, and reference-image
# editing with the Instruct checkpoint. The transformer shards with tensor
# parallelism over the 8 cores (~20 GB of weights per core); it has no
# sequence-parallel or CFG-parallel path.
#   ./hunyuan_image3.sh                                   # Instruct
#   MODEL=tencent/HunyuanImage-3.0 ./hunyuan_image3.sh    # base (text-to-image)
set -euo pipefail
export MODEL="${MODEL:-tencent/HunyuanImage-3.0-Instruct}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---tp-size ${NUM_GPUS}}"
export MODEL_ARGS="${MODEL_ARGS---dit-cpu-offload false}"
exec bash "$(dirname "$0")/launch.sh" "$@"
