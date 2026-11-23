#!/usr/bin/env bash
# FLUX.3 Image Klein text-to-image and image editing on a tpu7x 2x2x1 host:
# Ulysses over the cores (24 heads), the DiT blocks and the text encoder's
# layers compiled (sglang_tpu/multimodal_gen/flux3_image.py), every component
# resident, 1024x1024 compiled at startup.
#   ./flux3_image.sh
#   NUM_GPUS=1 ./flux3_image.sh
#   ./flux3_image.sh --batching-max-size 2 --batching-delay-ms 5
set -euo pipefail
export MODEL="${MODEL:-black-forest-labs/flux-3-image-klein}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree ${NUM_GPUS}}"
export MODEL_ARGS="${MODEL_ARGS---warmup-resolutions 1024x1024}"
# Under Ulysses, a 32 MiB scoped-VMEM limit instead of launch.sh's 64 MiB: the
# denoise is 15% faster at Ulysses 8. One core keeps 64 MiB (8% faster there).
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  case " ${LIBTPU_INIT_ARGS:-} " in
    *" --xla_tpu_scoped_vmem_limit_kib="*) ;;
    *) export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+${LIBTPU_INIT_ARGS} }--xla_tpu_scoped_vmem_limit_kib=32768" ;;
  esac
fi
exec bash "$(dirname "$0")/launch.sh" "$@"
