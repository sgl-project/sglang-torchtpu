#!/usr/bin/env bash
# Provision the TPU CI node pools. Prints every command; nothing runs without --apply.
#
#   ./provision-pools.sh              # preflight + print
#   ./provision-pools.sh --apply      # actually create
#
# Target layout (chips: tpu7x-standard-4t = 4 chips/host):
#
#   lane      hosts  pool                     ci-tpu-pool   nodes  chips
#   pr-gate   1      ci-tpu7x-2x2x1           pr-gate       4      16   (pre-existing)
#   pr-gate   2      ci-tpu7x-2x2x2-prgate    pr-gate-2h    2       8
#   nightly   1      ci-tpu7x-2x2x1-nightly   nightly-1h    2       8
#   nightly   2      ci-tpu7x-2x2x2           nightly       2       8   (pre-existing)
#   nightly   4      ci-tpu7x-2x2x4-nightly   nightly-4h    4      16
#                                             pr-gate total        24
#                                             nightly total        32
#
# Single-host pools carry no --tpu-topology: GKE derives the 2x2x1 node label from the machine
# type, and passing the flag takes the multi-host path, which implies a placement policy and
# fails ("...with placement policy is not supported. Use workload policy instead").
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
# ci-tpu7x-2x2x2 (ci-tpu-pool=nightly) already exists and serves nightly/2 -- not listed.
POOLS=(
  "ci-tpu7x-2x2x2-prgate   pr-gate-2h   2x2x2"
  "ci-tpu7x-2x2x1-nightly  nightly-1h   2x2x1"
  "ci-tpu7x-2x2x4-nightly  nightly-4h   2x2x4"
)

echo "== preflight (read-only) =="
echo "-- reservation headroom (CHIPS): reserved / in-use. Need 40 free."
# Aggregate reservation (VM_FAMILY_CLOUD_TPU_POD_SLICE_TPU7X), so the counts live under
# aggregateReservation, not specificReservation -- and they count chips, not VMs. Spot pools
# (tpu7x-2x2x1-i308, tpu7x-2x2x2-mimo) do not draw on it, so in-use < total TPU nodes.
gcloud compute reservations describe "${RESERVATION##*/}" \
  --zone="$ZONE" --project="$PROJECT" \
  --format="value(aggregateReservation.reservedResources[0].accelerator.acceleratorCount,aggregateReservation.inUseResources[0].accelerator.acceleratorCount)" || true
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
  if gcloud compute resource-policies describe "${name}-wp" --region="$REGION" \
       --project="$PROJECT" >/dev/null 2>&1; then
    echo "  (exists) ${name}-wp"
    continue
  fi
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
  state=$(gcloud container node-pools describe "$name" --cluster="$CLUSTER" --zone="$ZONE" \
            --project="$PROJECT" --format="value(status)" 2>/dev/null || true)
  case "$state" in
    RUNNING)      echo "  (exists) $name"; continue ;;
    PROVISIONING) echo "  (in progress) $name -- re-run once it is RUNNING"; continue ;;
    "")           ;;
    *)            echo "  ⚠️  $name is in state $state -- delete it before retrying:"
                  echo "      gcloud container node-pools delete $name --cluster=$CLUSTER --zone=$ZONE --project=$PROJECT"
                  continue ;;
  esac
  args=(
    gcloud container node-pools create "$name"
    --cluster="$CLUSTER" --zone="$ZONE" --project="$PROJECT"
    --machine-type="$MACHINE"
    --reservation-affinity=specific --reservation="$RESERVATION"
    --node-labels="ci-tpu-pool=${label}"
    --workload-metadata="$WORKLOAD_METADATA"
  )
  if [ "$topo" = "2x2x1" ]; then
    # Single-host: do NOT pass --tpu-topology. GKE derives the 2x2x1 node label from the
    # machine type, and passing the flag takes the multi-host path, which implies a
    # placement policy -- rejected as "Creation of a managed instance group with
    # tpu7x-standard-4t machine type with placement policy is not supported. Use workload
    # policy instead." The working ci-tpu7x-2x2x1 pool has no topology or placement field
    # at all, just machineType + node count.
    args+=(--num-nodes=2)
  else
    args+=(--tpu-topology="$topo" --placement-policy="${name}-wp")
  fi
  run "${args[@]}"
done
echo

echo "== 3. verify =="
echo "  kubectl get nodes -L ci-tpu-pool,cloud.google.com/gke-tpu-topology"
echo "  expect: 4 pr-gate, 2 pr-gate-2h, 2 nightly-1h, 2 nightly, 4 nightly-4h"
# ci-tpu7x-2x2x1 is deliberately left at 4 nodes. Growing it is safe whenever more single-host
# PR capacity is wanted (each 2x2x1 node is its own slice, so no straddle risk) -- just bump
# the tpu-prgate-1h nominalQuota in kueue-tpu.yaml to match:
#   gcloud container clusters resize tpu-cluster --node-pool=ci-tpu7x-2x2x1 --num-nodes=N ...
[ "$APPLY" = 1 ] || echo -e "\n(dry run -- nothing was created; re-run with --apply)"
