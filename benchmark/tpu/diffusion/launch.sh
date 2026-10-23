#!/usr/bin/env bash
# Shared launcher for the diffusion recipes in this directory. Model scripts set
# MODEL, PARALLEL_ARGS and MODEL_ARGS; each can be overridden from the
# environment (an empty value drops the default), and extra arguments are
# passed to the server.
set -euo pipefail

export SGLANG_PLATFORM="${SGLANG_PLATFORM:-tpu}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
# Highest TPU clock, and a 65536 KiB scoped-VMEM limit that gives XLA fusions
# the full 64 MiB of per-core VMEM on tpu7x (3% faster denoise on
# Z-Image-Turbo than without it), unless LIBTPU_INIT_ARGS already sets those
# flags.
for flag in --xla_tpu_dvfs_p_state=7 --xla_tpu_scoped_vmem_limit_kib=65536; do
  case " ${LIBTPU_INIT_ARGS:-} " in
    *" ${flag%%=*}="*) ;;
    *) LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+${LIBTPU_INIT_ARGS} }${flag}" ;;
  esac
done
export LIBTPU_INIT_ARGS

: "${MODEL:?set MODEL}"
NUM_GPUS="${NUM_GPUS:-8}"
PARALLEL_ARGS="${PARALLEL_ARGS-}"
MODEL_ARGS="${MODEL_ARGS-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"

# shellcheck disable=SC2086
exec python3 -m sglang.multimodal_gen.runtime.launch_server \
  --model-path "${MODEL}" \
  --num-gpus "${NUM_GPUS}" \
  ${PARALLEL_ARGS} \
  ${MODEL_ARGS} \
  --attention-backend tpu_splash_attn \
  --enable-torch-compile \
  --encoder-parallel fold \
  --text-encoder-cpu-offload false \
  --host "${HOST}" \
  --port "${PORT}" \
  "$@"
