#!/usr/bin/env bash
# Falcon sources must pin SOURCE_COMMIT and mount the model/private wheel assets.
set -euo pipefail
cd /workspace/sglang-torchtpu
DISCOVERY="$ARTIFACT_LOCAL_DIR/glm52-e2e/rank-$FALCON_RANK/discovery"
python3 -m benchmark.tpu.glm52_baseline.discover_checkpoint --output "$DISCOVERY"
export GLM52_MODEL_PATH
GLM52_MODEL_PATH=$(cat "$DISCOVERY/model-path.txt")
bash benchmark/tpu/glm52_baseline/bootstrap.sh
source /tmp/glm52-baseline-runtime/setup.sh
# Fail before allocating model weights if selective-reader regression tests fail.
SGLANG_PLATFORM=cpu "$PY" -m unittest discover -s test/registered/unit/hardware_backend/tpu -p test_ep_local_expert_reader.py -v > "$GLM52_ART/reader-tests.log" 2>&1
export GLM52_RUN_ID="${GLM52_RUN_ID:-${FALCON_EXP_ID:-baseline}}"
"$PY" -m benchmark.tpu.glm52_baseline.prepare --model "$GLM52_MODEL_PATH" --output /tmp/glm52-baseline-runtime/recipe
exec "$PY" -u -m benchmark.tpu.glm52_baseline.session
