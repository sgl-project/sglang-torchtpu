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
"$PY" -u test/registered/tpu/unit/test_dsa_index_topk.py TestDsaRadixTopk \
  2>&1 | tee "$OUT/device-tests.log"
"$PY" -u test/manual/kernels/benchmark_glm52_radix_topk.py --repeats 20 \
  --source-v2 benchmark/tpu/glm52_baseline/reference/radix_topk_v2.py \
  2>&1 | tee "$OUT/benchmark.jsonl"
echo TOPK_KERNEL_PASS
