#!/usr/bin/env bash
# Kandinsky 6.0 video super-resolution (VSR-distilled2steps, x2 / x4 / x2.25) on one tpu7x 2x2x1 host
# (8 cores): Ulysses 4 (28 heads) x ring 2, per-block torch.compile, whole tiles decoded across the
# cores. The source audio is kept.
#   ./kandinsky6_sr.sh
#   MODEL=kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers ./kandinsky6_sr.sh
set -euo pipefail
export MODEL="${MODEL:-kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree 4 --ring-degree 2}"
export MODEL_ARGS="${MODEL_ARGS---dit-cpu-offload false --vae-config.use-parallel-tiling true}"
exec bash "$(dirname "$0")/launch.sh" --regional-compile "$@"
