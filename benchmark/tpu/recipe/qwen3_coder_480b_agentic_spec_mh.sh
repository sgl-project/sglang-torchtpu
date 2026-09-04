#!/usr/bin/env bash
# Multi-host CI recipe: Qwen3-Coder-480B-A35B-Instruct-FP8 colocated agentic
# EAGLE3 server -- tp=16, ep=16, dp=2 -> attention_tp = 16/2 = 8, across 2 hosts.
#
# Declared by test/registered/tpu/e2e/
# test_tpu_qwen3_coder_480b_multihost_agentic_spec.py via MULTIHOST_LAUNCH; the
# TPU CI JobSet runs this on every host and rank 0 runs that test against the
# resulting server.
set -uo pipefail
cd "$(dirname "$0")/.."   # recipe/ -> benchmark/tpu

PROFILE=${PROFILE:-ci}

# The target checkpoint otherwise remains in the host page cache after loading.
# Together with the pinned HiCache pool and compiler state, that can exhaust host
# memory while TPU HBM still has headroom.
EXTRA_ARGS="--weight-loader-drop-cache-after-load${EXTRA_ARGS:+ $EXTRA_ARGS}"

# The CI sweep reaches 16 concurrent requests globally. With DP=2 that is eight
# requests per attention-DP worker, so compile only the reachable batch buckets.
if [[ "$PROFILE" == "ci" ]]; then
  export MAX_RUNNING="${MAX_RUNNING:-8}"
  export TORCH_COMPILE_MAX_BS="${TORCH_COMPILE_MAX_BS:-8}"
  export TPU_COMPILE_TOKENS="${TPU_COMPILE_TOKENS:-4096}"
  export TPU_COMPILE_BS="${TPU_COMPILE_BS:-1 8}"
fi

exec env \
  MODEL="${MODEL:-/weights/Qwen3-Coder-480B-A35B-Instruct-FP8}" \
  DRAFT_MODEL="${DRAFT_MODEL:-/weights/SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-SpecForge-EigenAI}" \
  PROFILE="$PROFILE" \
  EXTRA_ARGS="$EXTRA_ARGS" \
  ROLE=colocated \
  bash recipe/qwen3_coder_480b/nightly/serve.sh
