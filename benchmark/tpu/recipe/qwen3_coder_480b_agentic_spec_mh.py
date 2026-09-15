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

# Simulate mean accept length 2.5, including the bonus, on synthetic prompts.
# Fixed token output keeps this a synthetic throughput benchmark.
export SGLANG_SIMULATE_ACC_LEN=2.5
export SGLANG_SIMULATE_ACC_METHOD=match-expected
export SGLANG_SIMULATE_ACC_TOKEN_MODE=fixed
echo "[480b-agentic-spec] simulated accept_length=2.5 (includes bonus), method=$SGLANG_SIMULATE_ACC_METHOD, token_mode=$SGLANG_SIMULATE_ACC_TOKEN_MODE"

# Capacity and compile buckets come from the same config as the baseline.
exec env \
  MODEL="${MODEL:-/weights/Qwen3-Coder-480B-A35B-Instruct-FP8}" \
  DRAFT_MODEL="${DRAFT_MODEL:-/weights/SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-SpecForge-EigenAI}" \
  PROFILE="$PROFILE" \
  ROLE=colocated \
  bash recipe/qwen3_coder_480b/nightly/serve.sh
