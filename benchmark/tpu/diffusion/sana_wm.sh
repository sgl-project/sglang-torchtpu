#!/usr/bin/env bash
# SANA-WM camera-controlled image-to-video on one TensorCore, every component
# resident: chunk-causal streaming stage 1 and the chunked LTX-2 refiner.
#   ./sana_wm.sh
set -euo pipefail
export MODEL="${MODEL:-Efficient-Large-Model/SANA-WM_streaming}"
export NUM_GPUS="${NUM_GPUS:-1}"
export PARALLEL_ARGS="${PARALLEL_ARGS-}"
export MODEL_ARGS="${MODEL_ARGS---streaming --refiner-chunked --regional-compile --dit-cpu-offload false --dit-layerwise-offload false --warmup-mode request}"
# A 32 MiB scoped-VMEM limit instead of launch.sh's 64 MiB: at 64 MiB the compiled
# stage-1 blocks of a 3-frame chunk return NaN on tpu7x.
case " ${LIBTPU_INIT_ARGS:-} " in
  *" --xla_tpu_scoped_vmem_limit_kib="*) ;;
  *) export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+${LIBTPU_INIT_ARGS} }--xla_tpu_scoped_vmem_limit_kib=32768" ;;
esac
exec bash "$(dirname "$0")/launch.sh" "$@"
