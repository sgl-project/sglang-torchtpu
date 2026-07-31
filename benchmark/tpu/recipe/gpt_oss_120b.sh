#!/usr/bin/env bash
# Single-host launch recipe: gpt-oss-120b MXFP4 MoE (tp=8, ep=8) on one v7x host.
# This script IS the model+config spec; the nightly runs exactly it, so a manual repro
# of a CI failure is `./gpt_oss_120b.sh` and nothing else.
#
# --served-model-name MUST stay in sync with the eval client: the GPQA harness in
# test/registered/tpu/test_tpu_gpt_oss_120b.py passes model="gpt-oss-120b" and the
# request 404s if the served name differs.
#
# --reasoning-parser gpt-oss routes the Harmony analysis channel into
# reasoning_content, so the harness grades final-channel text only.
#
# SGLANG_TPU_FUSED_MOE_SC_GATHER is a backend env var, not a launch_server.sh knob:
# SC gather (PR #188) cannot lower gpt-oss's topk=4 + swigluoai fused-MoE, so keep the
# pre-#188 path until the kernel supports it.
#
# CONTEXT_LEN 16384 fits the longest GPQA prompt (~2.9k tokens) plus an 8k completion
# budget. The workload only prefills <=4096-token chunks and decodes at bs<=32, so one
# compile bucket each keeps bring-up ~10 min (validated 2026-07-28).
set -uo pipefail
cd "$(dirname "$0")/.."   # recipe/ -> benchmark/tpu
# QUANTIZATION= / KV_CACHE_DTYPE= (empty): defeat launch_server.sh's fp8 defaults.
# MXFP4 is auto-detected from the checkpoint and the KV cache stays bf16.
exec env \
  MODEL="${MODEL:-/weights/gpt-oss-120b}" \
  TP_SIZE="${TP_SIZE:-8}" \
  EP_SIZE="${EP_SIZE:-8}" \
  QUANTIZATION= \
  KV_CACHE_DTYPE= \
  SGLANG_TPU_FUSED_MOE_SC_GATHER="${SGLANG_TPU_FUSED_MOE_SC_GATHER:-0}" \
  EXTRA_ARGS="--reasoning-parser gpt-oss --served-model-name gpt-oss-120b${EXTRA_ARGS:+ $EXTRA_ARGS}" \
  MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.60}" \
  CONTEXT_LEN="${CONTEXT_LEN:-16384}" \
  TPU_COMPILE_TOKENS="${TPU_COMPILE_TOKENS:-4096}" \
  TPU_COMPILE_BS="${TPU_COMPILE_BS:-32}" \
  MAX_RUNNING="${MAX_RUNNING:-32}" \
  TORCH_COMPILE_MAX_BS="${TORCH_COMPILE_MAX_BS:-32}" \
  ENABLE_PROFILER_ENDPOINT="${ENABLE_PROFILER_ENDPOINT:-0}" \
  bash launch_server.sh
