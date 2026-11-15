#!/usr/bin/env bash
# One baseline and one retained-metadata sequence; no checkpoint or model.
set -euo pipefail
ulimit -c 0
: "${SOURCE_COMMIT:?}" "${ARTIFACT_LOCAL_DIR:?}" "${FALCON_RANK:?}"
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
OUT="$ARTIFACT_LOCAL_DIR/allocator-metadata-pair/rank-$FALCON_RANK"
mkdir -p "$OUT"
export LIBTPU_INIT_ARGS
LIBTPU_INIT_ARGS=$(python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('benchmark/tpu/glm52_epic/torch_runnable_baseline.json').read_text())['runtime_env']['LIBTPU_INIT_ARGS'])
PY
)
for arm in baseline metadata; do
  options=()
  [[ "$arm" != metadata ]] || options=(--preceding-metadata)
  timeout --signal=TERM --kill-after=15s 180s python -m benchmark.tpu.glm52_epic.repro_allocator_transfer \
    "${options[@]}" 2>&1 | tee "$OUT/$arm.log"
  python - "$OUT/$arm.log" "$arm" <<'PY'
from pathlib import Path
import sys
lines = Path(sys.argv[1]).read_text().splitlines()
assert lines.count('EAGER_MODE EagerMode.DEFER_NEVER') == 1
for prefix, count in ((0, 4), (0, 256), (3, 4), (127, 4)):
    for iteration in range(2):
        suffix = f'prefix={prefix} count={count} iteration={iteration}'
        assert lines.count('ALLOCATOR_TRANSFER_PASS ' + suffix) == 1
        assert lines.count('ALLOCATOR_METADATA_SEQUENCE_PASS ' + suffix) == int(sys.argv[2] == 'metadata')
print('ALLOCATOR_METADATA_PAIR_PASS arm=' + sys.argv[2], flush=True)
PY
done
