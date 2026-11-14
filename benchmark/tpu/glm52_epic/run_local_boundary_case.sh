#!/usr/bin/env bash
set -euo pipefail
case_name=${1:?case required}
unset RANK LOCAL_RANK LOCAL_WORLD_SIZE WORLD_SIZE
unset NNODES NODE_RANK MASTER_ADDR MASTER_PORT JOB_COMPLETION_INDEX
unset TORCH_TPU_SLICEBUILDER_ADDRESSES TORCH_TPU_TOPOLOGY
export TPU_WORKER_HOSTNAMES="$(hostname)"
discovery=$(python -m torch.tpu.distributed)
printf 'DISCOVERY_BEGIN\n%s\nDISCOVERY_END\n' "$discovery"
eval "$discovery"
export TORCH_TPU_TOPOLOGY TORCH_TPU_SLICEBUILDER_ADDRESSES WORLD_SIZE
[[ "$WORLD_SIZE" == 8 ]]
IFS=, read -r -a addresses <<< "$TORCH_TPU_SLICEBUILDER_ADDRESSES"
[[ "${#addresses[@]}" == 8 ]]
for address in "${addresses[@]}"; do
    host=${address%:*}
    [[ "$host" == localhost || "$host" == 127.0.0.1 || "$host" == "$(hostname)" ]]
done
[[ -n "$TORCH_TPU_TOPOLOGY" ]]
printf 'LOCAL_DISCOVERY_VALIDATED topology=%s addresses=%s\n' "$TORCH_TPU_TOPOLOGY" "$TORCH_TPU_SLICEBUILDER_ADDRESSES"
printf 'CHILD_LAUNCH_BEGIN case=%s\n' "$case_name"
python -u -m torch.distributed.run --standalone --nproc_per_node="$WORLD_SIZE" \
    --tee 3 --redirects 3 \
    benchmark/tpu/glm52_epic/repro_distributed_boundary.py --case "$case_name"
child_rc=$?
printf 'CHILD_LAUNCH_END case=%s status=%s\n' "$case_name" "$child_rc"
exit "$child_rc"
