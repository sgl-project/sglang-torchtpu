#!/usr/bin/env bash
# JoyAI-Image-Edit image editing on a tpu7x 2x2x1 host: Ulysses 8 over the cores (32 heads), serial
# CFG, the DiT blocks compiled (--regional-compile). The prompt rows pad to a multiple of 256
# (sglang_tpu/multimodal_gen/joy_image.py), so a new prompt and the negative prompt reuse the
# block program of their bucket. Edits go to /v1/images/edits with the image to edit as the
# multipart "image" file.
#   ./joyai_image_edit.sh
set -euo pipefail
export MODEL="${MODEL:-jdopensource/JoyAI-Image-Edit-Diffusers}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---cfg-parallel-size 1 --ulysses-degree ${NUM_GPUS}}"
export MODEL_ARGS="${MODEL_ARGS---regional-compile}"
exec bash "$(dirname "$0")/launch.sh" "$@"
