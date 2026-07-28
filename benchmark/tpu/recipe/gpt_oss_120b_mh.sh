#!/usr/bin/env bash
# Multi-host launch recipe: gpt-oss-120b (MXFP4 MoE, 128 experts / top-4, 36 layers).
# tp = hosts x chips (2 hosts -> tp=16), EP=8. Weights load from the ReadOnlyMany
# Hyperdisk mounted at /weights (the test declares the PVC via
# MULTIHOST_LAUNCH.weights_pvc; hydrate via .github/tpu-ci/hydrate-gptoss120b.yaml).
# This script IS the model+config spec; the CI runs it on every host. GKE injects
# TPU_WORKER_* so launch_server.sh auto-derives NNODES / rank.
set -uo pipefail
cd "$(dirname "$0")/.."   # recipe/ -> benchmark/tpu
# QUANTIZATION= / KV_CACHE_DTYPE= (empty): the checkpoint is mxfp4 and sglang
# auto-detects it (TPU path dequantizes the MoE weights); bf16 KV. Same known-good
# pattern as the 30B recipe -- launch_server.sh's fp8 defaults do not apply here.
# --reasoning-parser gpt-oss: harmony output is split so the analysis channel goes to
# reasoning_content and only final-channel text reaches lm_eval's exact-match parsing.
exec env \
  MODEL="${MODEL:-/weights/gpt-oss-120b}" \
  EP_SIZE="${EP_SIZE:-8}" \
  QUANTIZATION= \
  KV_CACHE_DTYPE= \
  EXTRA_ARGS="${EXTRA_ARGS:---reasoning-parser gpt-oss}" \
  bash launch_server.sh
