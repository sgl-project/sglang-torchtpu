#!/usr/bin/env bash
# Helios text-to-video at 384x640 on a tpu7x 2x2x1 host, 8 cores, Ulysses 8. Ulysses shards each
# chunk's current tokens (Helios-Base 8640; the Helios-Distilled pyramid levels 540, 2160 and 8640);
# the 2400 history tokens stay replicated on every core.
# Helios runs both CFG branches on every core, so CFG parallel would only duplicate work.
#   ./helios.sh                                  # Helios-Distilled
#   MODEL=BestWishYsh/Helios-Base ./helios.sh
set -euo pipefail
export MODEL="${MODEL:-BestWishYsh/Helios-Distilled}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---cfg-parallel-size 1 --ulysses-degree ${NUM_GPUS}}"
# --dit-cpu-offload false: Helios keeps the DiT resident only with 120 GB of free device memory,
# more than a core has, so without it every request copies the DiT in from the host.
# --warmup-resolutions: the default video warmup runs 832x480, a shape the recipe does not serve;
# its pyramid level-0 latent (15x26) does not divide into 2x2 patches, so the first step fails.
export MODEL_ARGS="${MODEL_ARGS---regional-compile --dit-cpu-offload false --warmup-resolutions 640x384}"
exec bash "$(dirname "$0")/launch.sh" "$@"
