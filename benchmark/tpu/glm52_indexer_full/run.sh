#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd /workspace/sglang-torchtpu
bash benchmark/tpu/glm52_baseline/bootstrap.sh
source /tmp/glm52-baseline-runtime/setup.sh
export PYTHONPATH="$PWD/python:$PWD" JAX_PLATFORMS=tpu OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
OUT="$ARTIFACT_LOCAL_DIR/rank-0/benchmark"
mkdir -p "$OUT" "$ARTIFACT_LOCAL_DIR/profiling"
"$PY" -m pip freeze > "$ARTIFACT_LOCAL_DIR/profiling/python-packages.txt"
env | sort > "$ARTIFACT_LOCAL_DIR/profiling/env.txt"
"$PY" -u benchmark/tpu/glm52_indexer_full/run.py --mode bench --out "$OUT/unprofiled" 2>&1 | tee "$OUT/bench.log"
cp "$OUT/unprofiled/metrics.jsonl" "$OUT/metrics.jsonl"
mkdir -p "$ARTIFACT_LOCAL_DIR/rank-0/compiler/llo"
export LIBTPU_INIT_ARGS="$LIBTPU_INIT_ARGS --xla_xprof_register_llo_debug_info=true --xla_mosaic_dump_to=$ARTIFACT_LOCAL_DIR/rank-0/compiler/llo"
"$PY" -u benchmark/tpu/glm52_indexer_full/run.py --mode profile --out "$OUT/profiled" 2>&1 | tee "$OUT/profile.log"
