#!/usr/bin/env bash
# Multi-host CI recipe: MiMo-V2.5-Pro (1T MoE, blockwise-fp8) DP-attention, 2 hosts.
#   tp=16, ep=16, dp=2  ->  attention_tp = tp/dp = 16/2 = 8
#
# This mirrors the E2E-validated recipe/mimo/launch_mimo_25_pro_multihost.sh (PR #155)
# -- keep the two in sync. It re-states that env rather than exec'ing it because the
# validated recipe hardcodes its prefill sizing; the CI variant differs only in:
#   * MODEL defaults to the ROX Hyperdisk mount (MULTIHOST_LAUNCH.weights_pvc;
#     hydrate via .github/tpu-ci/hydrate-mimo25pro.yaml)
#   * CHUNKED_PREFILL_SIZE / MAX_PREFILL_TOKENS raised 4096 -> 8192, and 8192 added
#     to TPU_COMPILE_TOKENS: a prefill shape outside the precompiled buckets takes
#     the recompile path that crashes torch_tpu (InductorError KeyError 'tpu').
#   * no NODE_RANK/RANK0_HOST plumbing: on GKE the rank and peer list auto-derive
#     from TPU_WORKER_ID / TPU_WORKER_HOSTNAMES, and HOST/PORT come from the JobSet.
# MEM_FRACTION_STATIC=0.7 is load-bearing: the 1T fp8 weights are a large fixed cost
# per device, and lower values yield a NEGATIVE KV budget ("Not enough memory").
set -uo pipefail
cd "$(dirname "$0")/.."   # recipe/ -> benchmark/tpu
exec env \
  MODEL="${MODEL:-/weights/MiMo-V2.5-Pro}" \
  NNODES=2 CHIPS_PER_HOST=8 DIST_PORT="${DIST_PORT:-10011}" \
  TP_SIZE=16 EP_SIZE="${EP_SIZE:-16}" \
  CONTEXT_LEN="${CONTEXT_LEN:-8192}" \
  MAX_RUNNING="${MAX_RUNNING:-8}" TORCH_COMPILE_MAX_BS="${TORCH_COMPILE_MAX_BS:-16}" \
  MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.7}" \
  CHUNKED_PREFILL_SIZE=8192 MAX_PREFILL_TOKENS=8192 \
  TPU_COMPILE_TOKENS="64 128 256 512 1024 2048 4096 8192" \
  JSON_MODEL_OVERRIDE='{"vision_config":null,"audio_config":null}' \
  EXTRA_ARGS="--dp-size 2 --enable-dp-attention --enable-dp-lm-head --trust-remote-code ${EXTRA_ARGS:-}" \
  bash launch_server.sh
