#!/usr/bin/env bash
# Benchmark-only bounded page64 decode attention discriminator; no model.
set -euo pipefail
ulimit -c 0
: "${SOURCE_COMMIT:?}" "${ARTIFACT_LOCAL_DIR:?}" "${FALCON_RANK:?}"
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
OUT="$ARTIFACT_LOCAL_DIR/decode-memory/rank-$FALCON_RANK"
mkdir -p "$OUT"
export LIBTPU_INIT_ARGS
LIBTPU_INIT_ARGS=$(python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('benchmark/tpu/glm52_epic/torch_runnable_baseline.json').read_text())['runtime_env']['LIBTPU_INIT_ARGS'])
PY
)
for mode in eager dynamo; do
  status=0
  timeout --signal=TERM --kill-after=20s 440s python -m benchmark.tpu.glm52_epic.compile_memory_case \
    --case decode-capacity --mode "$mode" --output "$OUT/$mode" --timeout 400 --interval 0.5 || status=$?
  python - "$OUT/$mode/native.log" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
if p.exists():
    print(p.read_text(), flush=True)
PY
  [[ "$status" == 0 ]] || exit "$status"
  # Keep the numerical/native evidence visible in Falcon's durable pod logs.
  python - "$OUT/$mode" "$mode" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
result = json.loads(root.joinpath('result.json').read_text())
assert result['returncode'] == 0 and result['timed_out'] is False
assert root.joinpath('memory.jsonl').stat().st_size > 0
print('DECODE_MEMORY_CASE_PASS mode=' + sys.argv[2], flush=True)
PY
done
