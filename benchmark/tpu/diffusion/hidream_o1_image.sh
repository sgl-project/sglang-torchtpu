#!/usr/bin/env bash
# HiDream-O1-Image text-to-image (8B pixel-space UiT, no VAE). Ulysses spans the cores (32 heads,
# 8 K/V heads) with both CFG branches in one batch, per-layer compiled regions, a resident DiT, and
# the default 2048x2048 compiled at startup.
#   ./hidream_o1_image.sh                 # 8 cores, Ulysses 8
#   NUM_GPUS=1 ./hidream_o1_image.sh
set -euo pipefail
export MODEL="${MODEL:-HiDream-ai/HiDream-O1-Image}"
export NUM_GPUS="${NUM_GPUS:-8}"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  export PARALLEL_ARGS="${PARALLEL_ARGS---cfg-parallel-size 1 --ulysses-degree ${NUM_GPUS}}"
fi
export MODEL_ARGS="${MODEL_ARGS---regional-compile --dit-cpu-offload false --warmup-resolutions 2048x2048}"
# A 32 MiB scoped-VMEM limit instead of launch.sh's 64 MiB: a faster denoise step.
[[ " ${LIBTPU_INIT_ARGS:-} " == *" --xla_tpu_scoped_vmem_limit_kib="* ]] ||
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+${LIBTPU_INIT_ARGS} }--xla_tpu_scoped_vmem_limit_kib=32768"
exec bash "$(dirname "$0")/launch.sh" "$@"
