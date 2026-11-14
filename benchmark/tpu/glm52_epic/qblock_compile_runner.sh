#!/usr/bin/env bash
# Model-free compile comparison only; no numerical or full-model acceptance.
set -euo pipefail
ulimit -c 0
: "${SOURCE_COMMIT:?}" "${ARTIFACT_LOCAL_DIR:?}" "${FALCON_RANK:?}"
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=no)" ]]
OUT="$ARTIFACT_LOCAL_DIR/qblock-compile/rank-$FALCON_RANK"
mkdir -p "$OUT"
# One independent PJRT client per host; do not join the bootstrap's Torch mesh.
unset TPU_WORKER_HOSTNAMES TPU_WORKER_ID TPU_PROCESS_ADDRESSES TPU_PROCESS_BOUNDS
unset TPU_CHIPS_PER_PROCESS_BOUNDS TPU_PROCESS_PORT TPU_VISIBLE_CHIPS
export JAX_PLATFORMS=tpu
failed=0
for qb in 128 32; do
  set +e
  timeout --signal=TERM --kill-after=15s 240s python -m benchmark.tpu.glm52_epic.repro_qblock_vmem_compile \
    --query-block "$qb" --pages-per-step 2 2>&1 | tee "$OUT/qb-$qb.log"
  status=("${PIPESTATUS[@]}")
  set -e
  rc=${status[0]}
  if [[ "${status[1]}" != 0 ]]; then exit "${status[1]}"; fi
  printf 'QBLOCK_COMPILE_EXIT query_block=%s exit_code=%s\n' "$qb" "$rc" | tee -a "$OUT/status.txt"
  # qb128 is an expected-failure control, but its diagnostic must be inspected.
  # A timeout or unrelated failure is not proof of the original VMEM error.
  if [[ "$qb" == 32 && "$rc" != 0 ]]; then failed=1; fi
done
exit "$failed"
