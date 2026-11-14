#!/usr/bin/env bash
# SenseNova-U1.5 text-to-image (sensenova/SenseNova-U1.5-8B-MoT). The pipeline
# runs on one core, eagerly, with its own attention (it rejects
# --attention-backend and torch.compile), so it skips launch.sh's server flags.
#   ./sensenova_u1.sh
set -euo pipefail
export SGLANG_PLATFORM="${SGLANG_PLATFORM:-tpu}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
# launch.sh's highest TPU clock and 65536 KiB scoped-VMEM limit, unless
# LIBTPU_INIT_ARGS already sets those flags.
for flag in --xla_tpu_dvfs_p_state=7 --xla_tpu_scoped_vmem_limit_kib=65536; do
  case " ${LIBTPU_INIT_ARGS:-} " in
    *" ${flag%%=*}="*) ;;
    *) LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+${LIBTPU_INIT_ARGS} }${flag}" ;;
  esac
done
export LIBTPU_INIT_ARGS
MODEL="${MODEL:-sensenova/SenseNova-U1.5-8B-MoT}"
exec python3 -m sglang.multimodal_gen.runtime.launch_server \
  --model-path "${MODEL}" \
  --num-gpus 1 \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-30000}" \
  "$@"
