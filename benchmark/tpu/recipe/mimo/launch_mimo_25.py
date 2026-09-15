#!/usr/bin/env bash
# SINGLE-HOST MiMo-V2.5 (310B) DP-attention on 1x tpu7x-8 (tp=8).
#   tp=8, ep=8, dp=2  ->  attention_tp = tp/dp = 8/2 = 4 = num_kv_heads (fused-qkv OK)
#   torch_compile_max_bs >= max_running(8) * dp_size(2) = 16  (DP decode-padding bucket)
#
#   ./launch_mimo_25.sh            # wait for "The server is fired up and ready to roll!"
#
# Defaults: 8k context, max_running=8. Override any knob via env, e.g.
#   MODEL=/local/MiMo-V2.5 CONTEXT_LEN=32768 MAX_RUNNING=4 ./launch_mimo_25.sh
#
# Loads the real checkpoint from MODEL (default: the HF repo id). For a fast
# weightless perf/plumbing smoke test, add EXTRA_ARGS="--load-format dummy"
# (random weights — timings real, generated tokens meaningless). Shared env
# (PYTHON, LIBTPU flags, FP8 defaults, ...) comes from ../../common_env.sh,
# sourced by launch_server.sh.
set -uo pipefail
cd "$(dirname "$0")/../.."   # -> benchmark/tpu (launch_server.sh + common_env.sh)

MODEL="${MODEL:-XiaomiMiMo/MiMo-V2.5}" \
TP_SIZE=8 EP_SIZE=8 \
CONTEXT_LEN="${CONTEXT_LEN:-8192}" \
MAX_RUNNING="${MAX_RUNNING:-8}" TORCH_COMPILE_MAX_BS="${TORCH_COMPILE_MAX_BS:-16}" \
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.5}" \
CHUNKED_PREFILL_SIZE=2048 MAX_PREFILL_TOKENS=2048 \
TPU_COMPILE_TOKENS="64 128 256 512 1024 2048" \
JSON_MODEL_OVERRIDE='{"vision_config":null,"audio_config":null}' \
EXTRA_ARGS="--dp-size 2 --enable-dp-attention --enable-dp-lm-head --trust-remote-code ${EXTRA_ARGS:-}" \
exec ./launch_server.sh
