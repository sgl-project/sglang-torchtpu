#!/usr/bin/env bash
# Multi-host CI recipe: MiMo-V2.5-Pro (1T MoE, blockwise-fp8) DP-attention, 2 hosts.
#   tp=16, ep=16, dp=2  ->  attention_tp = tp/dp = 16/2 = 8
#   torch_compile_max_bs >= max_running(8) * dp_size(2) = 16  (DP decode-padding bucket)
#
# attention_tp=8 requires the Pro config's num_key_value_heads AND
# swa_num_key_value_heads to both be multiples of 8. The 310B MiMo-V2.5 has
# num_key_value_heads=4 and needs a different parallelism.
#
# On GKE the rank and the peer list derive from TPU_WORKER_ID /
# TPU_WORKER_HOSTNAMES, so every worker runs this with no argument. By hand,
# start rank 0 first and wait for "Initializing TPU distributed runtime":
#   Host 0:  ./mimo_v25_pro_mh.sh
#   Host 1:  NODE_RANK=1 RANK0_HOST=<host0-internal-ip> ./mimo_v25_pro_mh.sh
#
# MODEL defaults to the ROX Hyperdisk mount (MULTIHOST_LAUNCH.weights_pvc, hydrated
# by .github/tpu-ci/hydrate-mimo25pro.yaml). A /gcs mount also needs
# EXTRA_ARGS="--weight-loader-disable-mmap": GCS-FUSE mmap SIGBUSes on these shards.
#
# MEM_FRACTION_STATIC=0.7 is load-bearing: the 1T fp8 weights are a large fixed
# cost per device, and lower values yield a NEGATIVE KV budget ("Not enough
# memory"). MAX_TOTAL_TOKENS is left to sglang, which sizes the KV pool from it.
#
# A prefill shape outside TPU_COMPILE_TOKENS takes the recompile path that
# crashes torch_tpu (InductorError KeyError 'tpu'), so 8192 is in both it and
# CHUNKED_PREFILL_SIZE / MAX_PREFILL_TOKENS.
set -euo pipefail
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
