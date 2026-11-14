#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
: "${SOURCE_COMMIT:?}" "${ARTIFACT_LOCAL_DIR:?}" "${FALCON_RANK:?}"
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
OUT="$ARTIFACT_LOCAL_DIR/qblock-numeric/rank-$FALCON_RANK"
mkdir -p "$OUT"
export LIBTPU_INIT_ARGS
LIBTPU_INIT_ARGS=$(python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('benchmark/tpu/glm52_epic/torch_runnable_baseline.json').read_text())['runtime_env']['LIBTPU_INIT_ARGS'])
PY
)
oracles=(self causal-mean)
if [[ "${GLM52_QBLOCK_ORACLE_SUITE:-standard}" == mixed ]]; then
  oracles=(mixed-self)
elif [[ "${GLM52_QBLOCK_ORACLE_SUITE:-standard}" != standard ]]; then
  echo 'Unsupported oracle suite' >&2
  exit 2
fi
for oracle in "${oracles[@]}"; do
  timeout --signal=TERM --kill-after=15s 300s python -m benchmark.tpu.glm52_epic.repro_qblock_torch_bridge \
    --query-block 32 --oracle "$oracle" 2>&1 | tee "$OUT/$oracle.log"
  python - "$OUT/$oracle.log" "$oracle" <<'PY'
from pathlib import Path
import sys
label = sys.argv[2].upper().replace('-', '_')
marker = f'QBLOCK_BRIDGE_{label}_PASS query_block=32'
assert Path(sys.argv[1]).read_text().splitlines().count(marker) == 1
PY
done
