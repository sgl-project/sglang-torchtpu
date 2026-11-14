#!/usr/bin/env bash
# Bounded production-op isolation, no checkpoint or model serving.
set -euo pipefail
ulimit -c 0
: "${SOURCE_COMMIT:?}" "${ARTIFACT_LOCAL_DIR:?}" "${FALCON_RANK:?}"
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
OUT="$ARTIFACT_LOCAL_DIR/sparse-step/rank-$FALCON_RANK"
mkdir -p "$OUT"
export LIBTPU_INIT_ARGS
LIBTPU_INIT_ARGS=$(python - <<'PY'
import json
from pathlib import Path
c=json.loads(Path('benchmark/tpu/glm52_epic/torch_runnable_baseline.json').read_text())
print(c['runtime_env']['LIBTPU_INIT_ARGS'])
PY
)
failed=0
for queries in 1 64; do
    set +e
    timeout --signal=TERM --kill-after=15s 600s python -m benchmark.tpu.glm52_epic.repro_sparse_step \
        --queries "$queries" > "$OUT/q$queries.log" 2>&1
    rc=$?
    set -e
    printf 'q%s exit_code=%s\n' "$queries" "$rc" | tee -a "$OUT/status.txt"
    tail -n 60 "$OUT/q$queries.log"
    if [[ "$rc" != 0 ]]; then failed=1; fi
done
exit "$failed"
