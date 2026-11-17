#!/usr/bin/env bash
set -euo pipefail
cd /workspace/sglang-torchtpu
bash benchmark/tpu/glm52_baseline/bootstrap.sh
source /tmp/glm52-baseline-runtime/setup.sh
export PYTHONPATH="$PWD/python:$PWD" JAX_PLATFORMS=tpu
OUT="$ARTIFACT_LOCAL_DIR/rank-0/benchmark"
mkdir -p "$OUT"
# One process without tracing instrumentation determines latency.
"$PY" -u benchmark/tpu/glm52_topk_profile/run.py --mode bench --out "$OUT/unprofiled" 2>&1 | tee "$OUT/bench.log"
# A fresh process collects diagnostic device timelines, never benchmark latency.
mkdir -p "$ARTIFACT_LOCAL_DIR/rank-0/compiler/llo"
export LIBTPU_INIT_ARGS="$LIBTPU_INIT_ARGS --xla_xprof_register_llo_debug_info=true --xla_mosaic_dump_to=$ARTIFACT_LOCAL_DIR/rank-0/compiler/llo"
"$PY" -u benchmark/tpu/glm52_topk_profile/run.py --mode profile --out "$OUT/profiled" 2>&1 | tee "$OUT/profile.log"
