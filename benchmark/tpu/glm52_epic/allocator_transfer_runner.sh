#!/usr/bin/env bash
# Production CPU page expansion and int32 roundtrip only; no checkpoint.
set -euo pipefail
ulimit -c 0
: "${SOURCE_COMMIT:?}" "${ARTIFACT_LOCAL_DIR:?}" "${FALCON_RANK:?}"
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
OUT="$ARTIFACT_LOCAL_DIR/allocator-transfer/rank-$FALCON_RANK"
mkdir -p "$OUT"
export LIBTPU_INIT_ARGS
LIBTPU_INIT_ARGS=$(python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('benchmark/tpu/glm52_epic/torch_runnable_baseline.json').read_text())['runtime_env']['LIBTPU_INIT_ARGS'])
PY
)
set +e
timeout --signal=TERM --kill-after=15s 180s python -m benchmark.tpu.glm52_epic.repro_allocator_transfer \
    2>&1 | tee "$OUT/transfer.log"
status=("${PIPESTATUS[@]}")
rc=${status[0]}
if [[ "$rc" == 0 && "${status[1]}" != 0 ]]; then rc=${status[1]}; fi
printf 'allocator-transfer exit_code=%s\n' "$rc" | tee "$OUT/status.txt"
exit "$rc"
