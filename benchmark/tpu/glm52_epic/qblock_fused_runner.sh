#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
: "${SOURCE_COMMIT:?}" "${ARTIFACT_LOCAL_DIR:?}" "${FALCON_RANK:?}"
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
OUT="$ARTIFACT_LOCAL_DIR/qblock-fused/rank-$FALCON_RANK"
mkdir -p "$OUT"
export LIBTPU_INIT_ARGS
LIBTPU_INIT_ARGS=$(python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('benchmark/tpu/glm52_epic/torch_runnable_baseline.json').read_text())['runtime_env']['LIBTPU_INIT_ARGS'])
PY
)
for selection in self causal-mean; do
  timeout --signal=TERM --kill-after=15s 400s python -m benchmark.tpu.glm52_epic.repro_sparse_step \
    --queries 2048 --entry prefill-qblock --selection "$selection" 2>&1 | tee "$OUT/$selection.log"
  python - "$OUT/$selection.log" "$selection" <<'PY'
from pathlib import Path
import sys
lines = Path(sys.argv[1]).read_text().splitlines()
for iteration in (0, 1):
    marker = (f'PRODUCTION_SPARSE_STEP_PASS iteration={iteration} queries=2048 '
              f'selection={sys.argv[2]} entry=prefill-qblock')
    assert lines.count(marker) == 1
PY
done
