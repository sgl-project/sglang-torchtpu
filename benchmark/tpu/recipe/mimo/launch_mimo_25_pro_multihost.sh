#!/usr/bin/env bash
# MULTI-HOST MiMo-V2.5-*Pro* (1T) DP-attention, TP=16 across 2x tpu7x-8 hosts.
#   tp=16, ep=16, dp=2  ->  attention_tp = tp/dp = 16/2 = 8
#   torch_compile_max_bs >= max_running(8) * dp_size(2) = 16  (DP decode-padding bucket)
#
# !!! PRECONDITION: attention_tp=8 requires the Pro config's num_key_value_heads
#     AND swa_num_key_value_heads to BOTH be multiples of 8. (The 310B MiMo-V2.5
#     has num_key_value_heads=4 and will FAIL this — use launch_mimo_25.sh for it.)
#
# RUN THE SAME COMMAND ON BOTH HOSTS, only NODE_RANK differs. Start rank 0 FIRST
# and let it reach "Initializing TPU distributed runtime" before starting rank 1:
#   Host 0 (rank 0, serves HTTP :30000):  ./launch_mimo_25_pro_multihost.sh 0
#   Host 1 (rank 1):   RANK0_HOST=<host0-internal-ip> ./launch_mimo_25_pro_multihost.sh 1
# GKE tpu7x-16 slice: run with NO arg on every worker (rank auto from TPU_WORKER_*).
#
# Defaults: 8k context, max_running=8. Override via env, e.g. the original 128k run:
#   CONTEXT_LEN=131072 MAX_RUNNING=2 ./launch_mimo_25_pro_multihost.sh 0
# (MAX_TOTAL_TOKENS is left to sglang — it auto-sizes the KV pool from the
#  mem-fraction budget, which with hybrid SWA holds far more than one context.)
#
# Loads the real 1T checkpoint from MODEL (default: the HF repo id). Stage it to
# the same path on both hosts; if loading from a /gcs mount add
# EXTRA_ARGS="--weight-loader-disable-mmap" (GCS-FUSE mmap SIGBUSes on large
# shards). For a fast weightless perf/plumbing smoke test add
# EXTRA_ARGS="--load-format dummy" (random weights — timings real, tokens
# meaningless). Shared env (PYTHON, LIBTPU flags, FP8 defaults, ...) comes from
# ../../common_env.sh, sourced by launch_server.sh.
set -uo pipefail
cd "$(dirname "$0")/../.."   # -> benchmark/tpu (launch_server.sh + common_env.sh)

NODE_RANK="${1:-${NODE_RANK:-}}"

# `exec env`: the conditional ${NODE_RANK:+...} / ${RANK0_HOST:+...} expand to whole
# words that bash won't treat as assignment prefixes — env parses them correctly.
# MEM_FRACTION_STATIC=0.7: the 1T fp8 weights are ~1/3 of each chip's HBM; a lower
# value (e.g. 0.3) yields a NEGATIVE KV budget ("Not enough memory").
exec env \
MODEL="${MODEL:-XiaomiMiMo/MiMo-V2.5-Pro}" \
NNODES=2 ${NODE_RANK:+NODE_RANK="$NODE_RANK"} ${RANK0_HOST:+RANK0_HOST="$RANK0_HOST"} \
CHIPS_PER_HOST=8 DIST_PORT="${DIST_PORT:-10011}" \
HOST=0.0.0.0 PORT="${PORT:-30000}" \
TP_SIZE=16 EP_SIZE="${EP_SIZE:-16}" \
CONTEXT_LEN="${CONTEXT_LEN:-8192}" \
MAX_RUNNING="${MAX_RUNNING:-8}" TORCH_COMPILE_MAX_BS="${TORCH_COMPILE_MAX_BS:-16}" \
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.7}" \
CHUNKED_PREFILL_SIZE=4096 MAX_PREFILL_TOKENS=4096 \
TPU_COMPILE_TOKENS="64 128 256 512 1024 2048 4096" \
JSON_MODEL_OVERRIDE='{"vision_config":null,"audio_config":null}' \
EXTRA_ARGS="--dp-size 2 --enable-dp-attention --enable-dp-lm-head --trust-remote-code ${EXTRA_ARGS:-}" \
./launch_server.sh
