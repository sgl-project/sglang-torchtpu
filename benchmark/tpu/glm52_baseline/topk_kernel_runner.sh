#!/usr/bin/env bash
# One bounded prod2 allocation; fail with the actual test/benchmark exit code.
set -euo pipefail
cd /workspace/sglang-torchtpu
bash benchmark/tpu/glm52_baseline/bootstrap.sh
source /tmp/glm52-baseline-runtime/setup.sh
OUT="$ARTIFACT_LOCAL_DIR/topk-kernel"
mkdir -p "$OUT"
export JAX_PLATFORMS=tpu PYTHONPATH=.:python
export JAX_COMPILATION_CACHE_DIR="/tmp/glm52-baseline-runtime/kernel-cache-$SOURCE_COMMIT"
export GLM52_RUNTIME_DIR="$OUT/actual-process-runtime"
export PYTHONPATH="$PWD/benchmark/tpu/glm52_baseline/runtime_probe:$PYTHONPATH"
"$PY" -u test/registered/kernels/ops/attention/test_dsa_radix_serving_op.py \
  2>&1 | tee "$OUT/serving-bridge.log"
JAX_PLATFORMS=tpu "$PY" -u test/manual/kernels/tune_glm52_radix_topk.py --repeats 12 \
  2>&1 | tee "$OUT/tuning.jsonl"
echo TOPK_TUNING_PASS
