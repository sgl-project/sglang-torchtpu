#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
: "${SOURCE_COMMIT:?}" "${ARTIFACT_LOCAL_DIR:?}" "${FALCON_RANK:?}"
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
OUT="$ARTIFACT_LOCAL_DIR/distributed-boundary/rank-$FALCON_RANK"
mkdir -p "$OUT"
export LIBTPU_INIT_ARGS
LIBTPU_INIT_ARGS=$(python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('benchmark/tpu/glm52_epic/torch_runnable_baseline.json').read_text())['runtime_env']['LIBTPU_INIT_ARGS'])
PY
)
failed=0
for case in roundtrip all-reduce asymmetric; do
    set +e
    timeout --signal=TERM --kill-after=15s 240s \
        bash benchmark/tpu/glm52_epic/run_local_boundary_case.sh "$case" \
        2>&1 | tee "$OUT/$case.log"
    status=("${PIPESTATUS[@]}")
    rc=${status[0]}
    if [[ "$rc" == 0 && "${status[1]}" != 0 ]]; then rc=${status[1]}; fi
    set -e
    printf '%s exit_code=%s\n' "$case" "$rc" | tee -a "$OUT/status.txt"
    pass_count=$(grep -c 'DISTRIBUTED_BOUNDARY_PASS' "$OUT/$case.log" || true)
    [[ "$rc" == 0 && "$pass_count" == 16 ]] || failed=1
    printf '%s pass_markers=%s\n' "$case" "$pass_count" | tee -a "$OUT/status.txt"
done
exit "$failed"
