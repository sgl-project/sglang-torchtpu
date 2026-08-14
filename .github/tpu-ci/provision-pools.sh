#!/usr/bin/env bash
# Provision the TPU CI node pools. Prints every command; nothing runs without --apply.
#
#   ./provision-pools.sh              # preflight + print
#   ./provision-pools.sh --apply      # actually create
#
# Target layout (chips: tpu7x-standard-4t = 4 chips/host):
#
#   lane      hosts  pool                     ci-tpu-pool   nodes  chips
#   pr-gate   1      ci-tpu7x-2x2x1           pr-gate       6      24   (resize 4 -> 6)
#   pr-gate   2      ci-tpu7x-2x2x2-prgate    pr-gate-2h    2       8
#   nightly   1      ci-tpu7x-2x2x1-nightly   nightly-1h    2       8
#   nightly   2      ci-tpu7x-2x2x2-nightly   nightly       2       8
#   nightly   4      ci-tpu7x-2x2x4-nightly   nightly-4h    4      16
#                                             pr-gate total        32
#                                             nightly total        32
#
# ONE POOL PER SLICE for multi-host: GKE tpu7x nodes expose no per-slice label, so
# ci-tpu-pool is the finest slice selector. Resizing a 2x2x2/2x2x4 pool puts two slices
# behind one label and a JobSet's pods can then straddle them -- no error, just an ICI
# rendezvous hang. Only the 2x2x1 pools are safe to resize, because each of their nodes
# is already a complete slice.
set -euo pipefail

PROJECT=sglang-torchtpu
ZONE=us-central1-c
REGION=us-central1
CLUSTER=tpu-cluster
MACHINE=tpu7x-standard-4t
RESERVATION="projects/${PROJECT}/reservations/ghostfish-u2zcmybntv70q"

# Match the existing CI pool. Cluster-level Workload Identity does NOT apply retroactively,
# and setting it after creation recreates every node in the pool -- so get it right here.
WORKLOAD_METADATA=GKE_METADATA

APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

run() {
  printf '  %s\n' "$*"
  [ "$APPLY" = 1 ] && "$@"
  return 0
}

# name  label  topology   (2x2x1 needs no placement policy; multi-host does)
POOLS=(
  "ci-tpu7x-2x2x2-prgate   pr-gate-2h   2x2x2"
  "ci-tpu7x-2x2x1-nightly  nightly-1h   2x2x1"
  "ci-tpu7x-2x2x2-nightly  nightly      2x2x2"
  "ci-tpu7x-2x2x4-nightly  nightly-4h   2x2x4"
)

echo "== preflight (read-only) =="
echo "-- reservation headroom: need 10 more ${MACHINE}"
gcloud compute reservations describe "${RESERVATION##*/}" \
  --zone="$ZONE" --project="$PROJECT" \
  --format="value(specificReservation.count,specificReservation.inUseCount)" || true
echo "-- 2x2x4 must be a valid tpu7x topology and be 4 hosts; confirm before creating"
echo "-- workload-metadata on the existing CI pool (this script assumes ${WORKLOAD_METADATA}):"
gcloud container node-pools describe ci-tpu7x-2x2x1 --cluster="$CLUSTER" \
  --zone="$ZONE" --project="$PROJECT" \
  --format="value(config.workloadMetadataConfig.mode)" || true
echo

echo "== 1. workload policies (multi-host only) =="
# --type must be HIGH_THROUGHPUT: COMPACT is the old node-pool --placement-type flag
# (different API), and HIGH_AVAILABILITY rejects --accelerator-topology.
for spec in "${POOLS[@]}"; do
  read -r name _label topo <<<"$spec"
  [ "$topo" = "2x2x1" ] && continue
  run gcloud compute resource-policies create workload-policy "${name}-wp" \
    --type=HIGH_THROUGHPUT --accelerator-topology="$topo" \
    --region="$REGION" --project="$PROJECT"
done
echo

echo "== 2. node pools =="
# --num-nodes is omitted deliberately: gcloud derives it from --tpu-topology for multi-host
# machine types. 2x2x1 is single-host, so it needs an explicit count.
for spec in "${POOLS[@]}"; do
  read -r name label topo <<<"$spec"
  args=(
    gcloud container node-pools create "$name"
    --cluster="$CLUSTER" --zone="$ZONE" --project="$PROJECT"
    --machine-type="$MACHINE" --tpu-topology="$topo"
    --reservation-affinity=specific --reservation="$RESERVATION"
    --node-labels="ci-tpu-pool=${label}"
    --workload-metadata="$WORKLOAD_METADATA"
  )
  if [ "$topo" = "2x2x1" ]; then
    args+=(--num-nodes=2)
  else
    args+=(--placement-policy="${name}-wp")
  fi
  run "${args[@]}"
done
echo

echo "== 3. grow the existing single-host pool 4 -> 6 =="
# Safe only because each 2x2x1 node is its own slice.
run gcloud container clusters resize "$CLUSTER" --node-pool=ci-tpu7x-2x2x1 \
  --num-nodes=6 --zone="$ZONE" --project="$PROJECT" --quiet
echo

echo "== 4. verify =="
echo "  kubectl get nodes -L ci-tpu-pool,cloud.google.com/gke-tpu-topology"
echo "  expect: 6 pr-gate, 2 pr-gate-2h, 2 nightly-1h, 2 nightly, 4 nightly-4h"
[ "$APPLY" = 1 ] || echo -e "\n(dry run -- nothing was created; re-run with --apply)"
