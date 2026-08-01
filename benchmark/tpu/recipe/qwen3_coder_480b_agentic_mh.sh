#!/usr/bin/env bash
# Multi-host CI recipe: Qwen3-Coder-480B-A35B-Instruct-FP8 colocated agentic
# benchmark server -- tp=16, ep=16, dp=4 -> attention_tp = 16/4 = 4, across 2 hosts.
#
# Declared by test/registered/tpu/test_tpu_qwen3_coder_480b_agentic.py via
# MULTIHOST_LAUNCH; the TPU CI JobSet runs this on EVERY host (one global TP group)
# and rank-0 then runs that test file against the resulting server. GKE injects
# TPU_WORKER_ID / TPU_WORKER_HOSTNAMES, so launch_server.sh auto-derives NNODES,
# node rank and the rendezvous address -- nothing to plumb here.
#
# This is a thin shim on purpose: every server knob lives in the benchmark's
# config.sh, so the CI server and a hand-run server are the same server. It only
# supplies what the CI environment owns:
#   * MODEL defaults to the ReadOnlyMany Hyperdisk mount (MULTIHOST_LAUNCH's
#     weights_pvc), not the HF repo id -- a 480 GB pull per run is not viable.
#   * PROFILE=ci sizes the sweep to the JobSet's deadline (see config.sh).
# HOST / PORT / ENABLE_PROFILER_ENDPOINT / ENABLE_COMPILE_CACHE come from the
# JobSet template.
#
# WEIGHTS PREREQUISITE: /weights/Qwen3-Coder-480B-A35B-Instruct-FP8 must exist on
# the model-weights-rox PVC. As of writing that disk carries MiMo-V2.5-Pro,
# Qwen3-30B-A3B-FP8 and gpt-oss-120b only, so this recipe needs a hydration pass
# before its first green run (same mechanism as the other multi-host models).
set -uo pipefail
cd "$(dirname "$0")/.."   # recipe/ -> benchmark/tpu

exec env \
  MODEL="${MODEL:-/weights/Qwen3-Coder-480B-A35B-Instruct-FP8}" \
  PROFILE="${PROFILE:-ci}" \
  ROLE=colocated \
  bash recipe/qwen3_coder_480b/nightly/serve.sh
