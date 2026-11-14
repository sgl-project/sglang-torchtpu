#!/usr/bin/env bash
# Intended local multi-device probe; discovery must be reviewed before launch.
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
    timeout --signal=TERM --kill-after=15s 240s bash -c '
        # The shared bootstrap intentionally exports a two-host list. This
        # probe must not inherit it: force the launcher onto this host only.
        unset RANK LOCAL_RANK LOCAL_WORLD_SIZE WORLD_SIZE
        unset NNODES NODE_RANK MASTER_ADDR MASTER_PORT JOB_COMPLETION_INDEX
        unset TORCH_TPU_SLICEBUILDER_ADDRESSES TORCH_TPU_TOPOLOGY
        export TPU_WORKER_HOSTNAMES="$(hostname)"
        discovery=$(python -m torch.tpu.distributed) || exit $?
        printf "DISCOVERY_BEGIN\n%s\nDISCOVERY_END\n" "$discovery"
        eval "$discovery"
        export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE
        [[ "$WORLD_SIZE" == 8 ]] || exit 2
        # Validate the launcher output in shell; do not embed a heredoc inside
        # the single-quoted nested bash command.
        IFS=, read -r -a addresses <<< "$TORCH_TPU_SLICEBUILDER_ADDRESSES"
        [[ "${#addresses[@]}" == 8 ]] || exit 2
        for address in "${addresses[@]}"; do
            host=${address%:*}
            [[ "$host" == localhost || "$host" == 127.0.0.1 || "$host" == "$(hostname)" ]] || exit 2
        done
        [[ -n "$TORCH_TPU_TOPOLOGY" ]] || exit 2
        printf 'LOCAL_DISCOVERY_VALIDATED topology=%s addresses=%s\n' "$TORCH_TPU_TOPOLOGY" "$TORCH_TPU_SLICEBUILDER_ADDRESSES"
        python -m torch.distributed.run --standalone --nproc_per_node="$WORLD_SIZE" \
            -m benchmark.tpu.glm52_epic.repro_distributed_boundary --case "$1"
    ' _ "$case" 2>&1 | tee "$OUT/$case.log"
    status=("${PIPESTATUS[@]}")
    rc=${status[0]}
    if [[ "$rc" == 0 && "${status[1]}" != 0 ]]; then rc=${status[1]}; fi
    set -e
    printf '%s exit_code=%s\n' "$case" "$rc" | tee -a "$OUT/status.txt"
    if [[ "$rc" != 0 ]]; then failed=1; fi
done
exit "$failed"
