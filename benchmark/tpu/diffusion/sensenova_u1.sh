#!/usr/bin/env bash
# SenseNova-U1.5 text-to-image (sensenova/SenseNova-U1.5-8B-MoT) on one core.
# The pipeline runs eagerly with its own attention: it takes neither
# --attention-backend nor torch.compile, so none of launch.sh's backend flags.
#   ./sensenova_u1.sh
set -euo pipefail
export MODEL="${MODEL:-sensenova/SenseNova-U1.5-8B-MoT}"
export NUM_GPUS="${NUM_GPUS:-1}"
export SERVER_ARGS="${SERVER_ARGS-}"
exec bash "$(dirname "$0")/launch.sh" "$@"
