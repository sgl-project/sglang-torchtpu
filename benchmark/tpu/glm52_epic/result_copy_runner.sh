#!/usr/bin/env bash
# Model-free result-copy fence A/B, independent process per case.
set -euo pipefail
ulimit -c 0
: "${SOURCE_COMMIT:?}" "${ARTIFACT_LOCAL_DIR:?}" "${FALCON_RANK:?}"
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
OUT="$ARTIFACT_LOCAL_DIR/result-copy/rank-$FALCON_RANK"
mkdir -p "$OUT"
export LIBTPU_INIT_ARGS
LIBTPU_INIT_ARGS=$(python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('benchmark/tpu/glm52_epic/torch_runnable_baseline.json').read_text())['runtime_env']['LIBTPU_INIT_ARGS'])
PY
)
failed=0
for producer in upload argmax; do
    for elements in 1 512; do
        for case in copy event copy-event; do
            name="${producer}-${elements}-${case}"
            set +e
            timeout --signal=TERM --kill-after=15s 120s python -m benchmark.tpu.glm52_epic.repro_result_copy \
                --producer "$producer" --elements "$elements" --case "$case" 2>&1 | tee "$OUT/$name.log"
            status=("${PIPESTATUS[@]}")
            rc=${status[0]}
            if [[ "$rc" == 0 && "${status[1]}" != 0 ]]; then rc=${status[1]}; fi
            set -e
            printf '%s exit_code=%s\n' "$name" "$rc" | tee -a "$OUT/status.txt"
            if [[ "$rc" != 0 ]]; then failed=1; fi
        done
    done
done
exit "$failed"
