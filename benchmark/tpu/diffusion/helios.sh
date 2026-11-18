#!/usr/bin/env bash
# Helios text-to-video at 384x640 on a tpu7x 2x2x1 host, 8 cores, Ulysses 8, both CFG branches on
# every core. Ulysses shards a chunk's current tokens; the 2400 history tokens stay replicated.
#   ./helios.sh                                  # Helios-Distilled
#   MODEL=BestWishYsh/Helios-Base ./helios.sh
set -euo pipefail
export MODEL="${MODEL:-BestWishYsh/Helios-Distilled}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---cfg-parallel-size 1 --ulysses-degree ${NUM_GPUS}}"
# --dit-cpu-offload false keeps the DiT resident (Helios asks for 120 GB free before it does).
# --warmup-resolutions: the default 832x480 warmup's level-0 latent does not split into 2x2 patches.
export MODEL_ARGS="${MODEL_ARGS---regional-compile --dit-cpu-offload false --warmup-resolutions 640x384}"
# A 32 MiB scoped-VMEM limit instead of launch.sh's 64 MiB: a faster denoise step.
[[ " ${LIBTPU_INIT_ARGS:-} " == *" --xla_tpu_scoped_vmem_limit_kib="* ]] ||
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+${LIBTPU_INIT_ARGS} }--xla_tpu_scoped_vmem_limit_kib=32768"
exec bash "$(dirname "$0")/launch.sh" "$@"
