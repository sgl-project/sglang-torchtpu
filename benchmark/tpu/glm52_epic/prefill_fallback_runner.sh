#!/usr/bin/env bash
# One production head-major fallback boundary; not full model/q-block compute.
set -euo pipefail
ulimit -c 0
: "${SOURCE_COMMIT:?}" "${ARTIFACT_LOCAL_DIR:?}" "${FALCON_RANK:?}"
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
OUT="$ARTIFACT_LOCAL_DIR/prefill-fallback/rank-$FALCON_RANK"
mkdir -p "$OUT"
export LIBTPU_INIT_ARGS
LIBTPU_INIT_ARGS=$(python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('benchmark/tpu/glm52_epic/torch_runnable_baseline.json').read_text())['runtime_env']['LIBTPU_INIT_ARGS'])
PY
)
set +e
timeout --signal=TERM --kill-after=15s 600s python -m benchmark.tpu.glm52_epic.repro_sparse_step \
    --queries 4 --entry prefill-fallback --selection self 2>&1 | tee "$OUT/query4.log"
status=("${PIPESTATUS[@]}")
rc=${status[0]}
if [[ "$rc" == 0 && "${status[1]}" != 0 ]]; then rc=${status[1]}; fi
printf 'query4 prefill-fallback exit_code=%s\n' "$rc" | tee "$OUT/status.txt"
exit "$rc"
