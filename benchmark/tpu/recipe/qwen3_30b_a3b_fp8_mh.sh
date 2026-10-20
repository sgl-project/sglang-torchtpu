#!/usr/bin/env bash
# Multi-host launch recipe: Qwen3-30B-A3B-FP8 MoE (tp = hosts x chips, EP=8). Weights
# load from the ReadOnlyMany Hyperdisk mounted at /weights (the test declares the PVC via
# MULTIHOST_LAUNCH.weights_pvc). This script IS the model+config spec; the CI runs it on
# every host. GKE injects TPU_WORKER_* so launch_server.sh auto-derives NNODES / rank.
# FP8 + MoE are auto-detected from the checkpoint; EP_SIZE must divide TP.
set -euo pipefail
cd "$(dirname "$0")/.."   # recipe/ -> benchmark/tpu
# QUANTIZATION= / KV_CACHE_DTYPE= (empty): the checkpoint is already block-fp8 and
# sglang auto-detects it, so let it. launch_server.sh's fp8 defaults (--quantization fp8
# --kv-cache-dtype fp8_e4m3) instead OOM the tp=16 compile (~144G HLO temporaries > 94G
# HBM). Empty = the known-good config validated at tp=16.
exec env \
  MODEL="${MODEL:-/weights/Qwen3-30B-A3B-FP8}" \
  EP_SIZE="${EP_SIZE:-8}" \
  QUANTIZATION= \
  KV_CACHE_DTYPE= \
  bash launch_server.sh
