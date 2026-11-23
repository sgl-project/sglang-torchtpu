#!/usr/bin/env bash
# Kandinsky 6.0 Pro-distill, text (or text and image) to video with audio, on one tpu7x 2x2x1 host
# (8 cores): Ulysses 8 (32 video heads), per-block torch.compile, all components resident.
#   ./kandinsky6.sh
set -euo pipefail
export MODEL="${MODEL:-kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree ${NUM_GPUS}}"
# --dit-cpu-offload false: the video defaults offload the DiT on TPU. --warmup-*: the default warmup
# runs 17 frames, a shape no request uses: warm up the served 768x512, 121-frame one.
export MODEL_ARGS="${MODEL_ARGS---dit-cpu-offload false --dit-layerwise-offload false --warmup-resolutions 768x512 --warmup-num-frames 121}"
# A 32 MiB scoped-VMEM limit instead of launch.sh's 64 MiB: a faster denoise step.
[[ " ${LIBTPU_INIT_ARGS:-} " == *" --xla_tpu_scoped_vmem_limit_kib="* ]] ||
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+${LIBTPU_INIT_ARGS} }--xla_tpu_scoped_vmem_limit_kib=32768"
exec bash "$(dirname "$0")/launch.sh" --regional-compile "$@"
