#!/usr/bin/env bash
# Bounded real TorchTPU EP8 bridge validation on one 4-chip host.
set -euo pipefail
cd /workspace/sglang-torchtpu
bash benchmark/tpu/glm52_baseline/bootstrap.sh
source /tmp/glm52-baseline-runtime/setup.sh
OUT="$ARTIFACT_LOCAL_DIR/moe-kernel"
mkdir -p "$OUT"
export PYTHONPATH="$PWD/python:$PWD"
export GLM52_RUNTIME_DIR="$OUT/actual-process-runtime"
export PYTHONPATH="$PWD/benchmark/tpu/glm52_baseline/runtime_probe:$PYTHONPATH"
export GLM_DIAGNOSTIC_DIR="$OUT/export-diagnostic"
"$PY" -u test/registered/kernels/ops/moe/test_glm_ep_moe_v2.py \
  2>&1 | tee "$OUT/device-tests.log"
echo MOE_PRODUCTION_SEQUENCE_PASS
