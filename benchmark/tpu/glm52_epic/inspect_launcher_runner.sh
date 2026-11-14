#!/usr/bin/env bash
# Read installed wheel Python source only; no TPU client or discovery execution.
set -euo pipefail
: "${SOURCE_COMMIT:?}" "${ARTIFACT_LOCAL_DIR:?}" "${FALCON_RANK:?}"
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
OUT="$ARTIFACT_LOCAL_DIR/launcher-inspection/rank-$FALCON_RANK"
mkdir -p "$OUT"
timeout --signal=TERM --kill-after=5s 30s python -m benchmark.tpu.glm52_epic.inspect_distributed_launcher \
    | tee "$OUT/source.txt"
