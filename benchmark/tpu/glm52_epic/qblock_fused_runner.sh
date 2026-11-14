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
compile_args=()
case "${GLM52_QBLOCK_COMPILE_MODE:-eager}" in
  eager) ;;
  dynamo) compile_args=(--dynamo) ;;
  *) echo 'Unsupported compile mode' >&2; exit 2 ;;
esac
selections=(self causal-mean)
case "${GLM52_QBLOCK_FUSED_SUITE:-standard}" in
  standard) ;;
  rope) selections=(rope-pair) ;;
  prefix) selections=(prefix-pair) ;;
  *) echo 'Unsupported fused suite' >&2; exit 2 ;;
esac
entry="${GLM52_SPARSE_PREFILL_ENTRY:-prefill-qblock}"
case "$entry" in
  prefill-qblock|prefill-long-table) ;;
  *) echo 'Unsupported prefill entry' >&2; exit 2 ;;
esac
prefix="${GLM52_SPARSE_PREFIX_TOKENS:-0}"
case "$prefix" in
  0) ;;
  49152) [[ "$entry" == prefill-long-table ]] ;;
  *) echo 'Unsupported sparse prefix' >&2; exit 2 ;;
esac
requests="${GLM52_SPARSE_REQUESTS:-1}"
case "$requests" in
  1) ;;
  2) [[ "$entry" == prefill-long-table && "$prefix" == 49152 && "${GLM52_QBLOCK_FUSED_SUITE:-standard}" == prefix ]] ;;
  *) echo 'Unsupported request count' >&2; exit 2 ;;
esac
for selection in "${selections[@]}"; do
  timeout --signal=TERM --kill-after=15s 400s python -m benchmark.tpu.glm52_epic.repro_sparse_step \
    --queries 2048 --entry "$entry" --prefix-tokens "$prefix" --requests "$requests" --selection "$selection" "${compile_args[@]}" 2>&1 | tee "$OUT/$selection.log"
  python - "$OUT/$selection.log" "$selection" "${GLM52_QBLOCK_COMPILE_MODE:-eager}" "$entry" "$prefix" "$requests" <<'PY'
from pathlib import Path
import sys
lines = Path(sys.argv[1]).read_text().splitlines()
requests = int(sys.argv[6])
assert lines.count(f'SPARSE_REQUESTS count={requests} queries_per_request={2048 // requests}') == 1
if sys.argv[4] == 'prefill-long-table':
    prefix = int(sys.argv[5])
    metadata = (f'SPARSE_METADATA pages_per_seq=481 live_kv={prefix + 2048 // requests} '
                f'prefix={prefix} expected_prefill_path=per-query')
    assert lines.count(metadata) == 1
if sys.argv[3] == 'dynamo':
    assert lines.count('PRODUCTION_SPARSE_DYNAMO_ENABLED fullgraph=True dynamic=False') == 1
for iteration in (0, 1):
    marker = (f'PRODUCTION_SPARSE_STEP_PASS iteration={iteration} queries=2048 '
              f'selection={sys.argv[2]} entry={sys.argv[4]}')
    assert lines.count(marker) == 1
PY
done
