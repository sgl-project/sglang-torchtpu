#!/usr/bin/env bash
# Hunyuan3D-2 image-to-mesh, shape generation (tencent/Hunyuan3D-2, DiT v2.0,
# fp16); texture painting is off (paint_enable=false). The DiT and the DINOv2
# conditioner run on every core (CFG parallel off); the volume decode splits
# its grid chunks over the cores, and rank 0 writes the mesh.
#   ./hunyuan3d.sh                 # 8 cores
#   NUM_GPUS=1 ./hunyuan3d.sh
set -euo pipefail
export MODEL="${MODEL:-tencent/Hunyuan3D-2}"
export NUM_GPUS="${NUM_GPUS:-8}"
shape_only="$(mktemp --suffix=.json "${TMPDIR:-/tmp}/hunyuan3d_shape.XXXXXX")"
echo '{"paint_enable": false}' >"${shape_only}"
export PARALLEL_ARGS="${PARALLEL_ARGS---cfg-parallel-size 1}"
export MODEL_ARGS="${MODEL_ARGS---regional-compile --dit-cpu-offload false --image-encoder-cpu-offload false --pipeline-config-path ${shape_only}}"
exec bash "$(dirname "$0")/launch.sh" "$@"
