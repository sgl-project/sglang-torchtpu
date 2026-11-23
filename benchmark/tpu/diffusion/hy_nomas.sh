#!/usr/bin/env bash
# HY-Nomas event-conditioned video world model on tpu7x: Ulysses over the 16 DiT heads, every component resident.
# One server answers /v1/videos jobs and ws://HOST:PORT/v1/realtime_video/generate sessions; the first request of
# a new video size or event length compiles. Ulysses 4 takes every event length at 352x640 (220 tokens a latent
# frame); Ulysses 8 needs an even latent count per event there.
#   ./hy_nomas.sh                  # 4 cores, Ulysses 4
#   NUM_GPUS=1 ./hy_nomas.sh
set -euo pipefail
export MODEL="${MODEL:-tencent/Hy-Nomas}"
export NUM_GPUS="${NUM_GPUS:-4}"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  export PARALLEL_ARGS="${PARALLEL_ARGS---sp-degree ${NUM_GPUS}}"
fi
if [[ "${NUM_GPUS}" -gt 1 && "${NUM_GPUS}" -lt 8 ]]; then
  # A server on part of the host's cores.
  export TORCH_TPU_DEVICES_PER_HOST="${TORCH_TPU_DEVICES_PER_HOST:-${NUM_GPUS}}"
fi
export MODEL_ARGS="${MODEL_ARGS---dit-cpu-offload false --dit-layerwise-offload false}"
# libtpu's 32 MiB scoped-VMEM default instead of launch.sh's 64 MiB: a 16% faster denoise step at Ulysses 4.
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_scoped_vmem_limit_kib=32768}"
exec bash "$(dirname "$0")/launch.sh" "$@"
