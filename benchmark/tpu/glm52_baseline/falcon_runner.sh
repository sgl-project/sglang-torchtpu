#!/usr/bin/env bash
# Falcon sources must pin SOURCE_COMMIT and mount the model/private wheel assets.
set -euo pipefail
cd /workspace/sglang-torchtpu
bash benchmark/tpu/glm52_baseline/bootstrap.sh
source /tmp/glm52-baseline-runtime/setup.sh
export GLM52_RUN_ID="${FALCON_EXP_ID:-baseline}"
"$PY" -m benchmark.tpu.glm52_baseline.prepare --output /tmp/glm52-baseline-runtime/recipe
exec "$PY" -u -m benchmark.tpu.glm52_baseline.session
