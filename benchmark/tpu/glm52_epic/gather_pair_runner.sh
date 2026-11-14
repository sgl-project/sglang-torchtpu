#!/usr/bin/env bash
# Checkpoint-free diagnostic; each host uses one device, not a model baseline.
set -euo pipefail
ulimit -c 0
: "${ARTIFACT_LOCAL_DIR:?}" "${FALCON_RANK:?}" "${SOURCE_COMMIT:?}"
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
OUT="$ARTIFACT_LOCAL_DIR/gather-pair/rank-$FALCON_RANK"
mkdir -p "$OUT"
# Use the exact short-baseline compiler settings, before importing either runtime.
export LIBTPU_INIT_ARGS
LIBTPU_INIT_ARGS=$(python - <<'PY'
import json
from pathlib import Path
c=json.loads(Path('benchmark/tpu/glm52_epic/torch_runnable_baseline.json').read_text())
print(c.get('runtime_env',c.get('runtime_envs',{}))['LIBTPU_INIT_ARGS'])
PY
)
# Never initialize a distributed client: these are independent single-device cases.
# A fatal in one case must not suppress evidence from the other backend.
failed=0
for queries in 1 64; do
for backend in jax torch; do
    case_id="$backend-q$queries"
    set +e
    timeout --signal=TERM --kill-after=15s 600s python -m benchmark.tpu.glm52_epic.repro_sparse_gather \
        --backend "$backend" --queries "$queries" --page-size 128 \
        --evidence-dir "$OUT/$case_id" > "$OUT/$case_id.log" 2>&1
    rc=$?
    set -e
    printf '%s exit_code=%s\n' "$case_id" "$rc" | tee -a "$OUT/status.txt"
    tail -n 30 "$OUT/$case_id.log"
    if [[ "$rc" != 0 ]]; then failed=1; fi
done
done
exit "$failed"
