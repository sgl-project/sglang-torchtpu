#!/usr/bin/env bash
# MOSS-Image text-to-image (OpenMOSS-Team/MOSS-Image-Base or -SFT) on a tpu7x 2x2x1 host:
# Ulysses 8 over the cores (24 heads), serial CFG, the joint blocks compiled
# (--regional-compile). The joint rows pad to a multiple of 256
# (sglang_tpu/multimodal_gen/moss_image.py), so a new prompt and the negative prompt reuse
# the block program of their bucket.
#   ./moss_image.sh
#   MODEL=OpenMOSS-Team/MOSS-Image-SFT ./moss_image.sh
#   NUM_GPUS=1 ./moss_image.sh
#   ./moss_image.sh --batching-max-size 4 --batching-delay-ms 20
set -euo pipefail
export MODEL="${MODEL:-OpenMOSS-Team/MOSS-Image-Base}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---cfg-parallel-size 1 --ulysses-degree ${NUM_GPUS}}"
# The DiT stays resident and whole on every core. The server warmup compiles the
# default 1024x1024 (its own default caps the warmup at 768x768).
export MODEL_ARGS="${MODEL_ARGS---regional-compile --dit-cpu-offload false --use-fsdp-inference false --warmup-resolutions 1024x1024}"
# Under Ulysses, a 32 MiB scoped-VMEM limit instead of launch.sh's 64 MiB: the denoise
# step is 13% faster on tpu7x (0.054 s against 0.062 s at Ulysses 8). One core keeps 64 MiB.
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  case " ${LIBTPU_INIT_ARGS:-} " in
    *" --xla_tpu_scoped_vmem_limit_kib="*) ;;
    *) export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+${LIBTPU_INIT_ARGS} }--xla_tpu_scoped_vmem_limit_kib=32768" ;;
  esac
fi
exec bash "$(dirname "$0")/launch.sh" "$@"
