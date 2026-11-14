#!/usr/bin/env bash
# Fresh processes isolate decode/extend and XLA/Pallas key writers.
set -euo pipefail
ulimit -c 0
: "${SOURCE_COMMIT:?}" "${ARTIFACT_LOCAL_DIR:?}" "${FALCON_RANK:?}"
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
OUT="$ARTIFACT_LOCAL_DIR/indexer-bridge/rank-$FALCON_RANK"
mkdir -p "$OUT"
export LIBTPU_INIT_ARGS
LIBTPU_INIT_ARGS=$(python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('benchmark/tpu/glm52_epic/torch_runnable_baseline.json').read_text())['runtime_env']['LIBTPU_INIT_ARGS'])
PY
)
failed=0
for queries in 1 4; do
    for writer in xla pallas; do
        set +e
        timeout --signal=TERM --kill-after=15s 300s python -m benchmark.tpu.glm52_epic.repro_sparse_indexer \
            --queries "$queries" --key-store "$writer" --segments serving 2>&1 | tee "$OUT/q${queries}-${writer}.log"
        status=("${PIPESTATUS[@]}")
        rc=${status[0]}
        if [[ "$rc" == 0 && "${status[1]}" != 0 ]]; then rc=${status[1]}; fi
        set -e
        printf 'queries=%s writer=%s exit_code=%s\n' "$queries" "$writer" "$rc" | tee -a "$OUT/status.txt"
        if [[ "$rc" != 0 ]]; then failed=1; fi
    done
done
exit "$failed"
