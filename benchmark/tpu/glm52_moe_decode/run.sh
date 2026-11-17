#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd /workspace/sglang-torchtpu
bash benchmark/tpu/glm52_baseline/bootstrap.sh
source /tmp/glm52-baseline-runtime/setup.sh
export PYTHONPATH="$PWD/python:$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT="$ARTIFACT_LOCAL_DIR/host-$FALCON_RANK"
mkdir -p "$OUT"
"$PY" -u benchmark/tpu/glm52_moe_decode/session.py 2>&1 | tee "$OUT/benchmark.log"
