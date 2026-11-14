#!/usr/bin/env bash
# Use the successful standalone TorchTPU ownership pattern, not native JAX TPU.
set -euo pipefail
ulimit -c 0
: "${SOURCE_COMMIT:?}" "${ARTIFACT_LOCAL_DIR:?}" "${FALCON_RANK:?}"
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
OUT="$ARTIFACT_LOCAL_DIR/qblock-bridge/rank-$FALCON_RANK"
mkdir -p "$OUT"
# Keep bootstrap TPU host metadata: removing it broke native JAX initialization.
export LIBTPU_INIT_ARGS
LIBTPU_INIT_ARGS=$(python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('benchmark/tpu/glm52_epic/torch_runnable_baseline.json').read_text())['runtime_env']['LIBTPU_INIT_ARGS'])
PY
)
failed=0
for qb in 128 32; do
  set +e
  timeout --signal=TERM --kill-after=15s 300s python -m benchmark.tpu.glm52_epic.repro_qblock_torch_bridge \
    --query-block "$qb" 2>&1 | tee "$OUT/qb-$qb.log"
  status=("${PIPESTATUS[@]}")
  set -e
  rc=${status[0]}
  if [[ "${status[1]}" != 0 ]]; then exit "${status[1]}"; fi
  printf 'QBLOCK_BRIDGE_EXIT query_block=%s exit_code=%s\n' "$qb" "$rc" | tee -a "$OUT/status.txt"
  if ! python - "$OUT/qb-$qb.log" "$qb" "$rc" <<'PY'
from pathlib import Path
import sys
text = Path(sys.argv[1]).read_text(errors='replace')
qb, rc = map(int, sys.argv[2:])
if qb == 128:
    valid = (rc == 1 and 'sparse_mla_prefill_qblock_tokens' in text
             and 'Ran out of memory in memory space vmem' in text)
else:
    marker = f'QBLOCK_BRIDGE_ZERO_PASS query_block={qb}'
    valid = rc == 0 and text.splitlines().count(marker) == 1
print(f'QBLOCK_BRIDGE_ACCEPT query_block={qb} valid={valid}', flush=True)
sys.exit(0 if valid else 1)
PY
  then failed=1; fi
done
exit "$failed"
