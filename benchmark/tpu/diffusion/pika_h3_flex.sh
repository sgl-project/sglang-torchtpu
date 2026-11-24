#!/usr/bin/env bash
# Pika-H3-Flex (MiniMax-H3 with Pika's six-step LoRA adapters, kept unmerged) video with audio on one tpu7x 2x2x1
# host (8 cores): Ulysses 8 over a replicated DiT, each DiT block compiled as one program with its LoRA matmuls.
# The VAEs stay on the TPU. The text encoder does too for t2va and fl2va; ref2va's long packed sequence (two
# 2048-pixel images and a video) needs its HBM, so there it stays on the host.
#   ./pika_h3_flex.sh                          # t2va, fl2va
#   ./pika_h3_flex.sh --model-variant ref2va   # ref2va
set -euo pipefail
export MODEL="${MODEL:-Pika-Labs/Pika-H3-Flex}"
export NUM_GPUS="${NUM_GPUS:-8}"
export PARALLEL_ARGS="${PARALLEL_ARGS---ulysses-degree 8}"
export MODEL_ARGS="${MODEL_ARGS---use-fsdp-inference false --dit-cpu-offload false --regional-compile --performance-mode manual --warmup-mode off}"
text_encoder_offload=false
case " $* " in *" ref2va "* | *"=ref2va "*) text_encoder_offload=true ;; esac
exec bash "$(dirname "$0")/launch.sh" --vae-cpu-offload false --text-encoder-cpu-offload "${text_encoder_offload}" "$@"
