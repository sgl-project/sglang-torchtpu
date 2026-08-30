#!/usr/bin/env python3
"""Resolve the node pools serving a (lane, host count), discovering new slices live.

topology.yaml lists the pools that existed when it was last edited. This unions that list
with every pool found on the cluster: a node labelled ci-tpu-lane=<lane> whose topology
spans <hosts> nodes. Adding a slice becomes "create the pool with ci-tpu-lane and
ci-tpu-pool labels" -- no edit to topology.yaml or kueue-tpu.yaml.

UNION, not replace, and that is the load-bearing choice. Pools predating this script carry
no ci-tpu-lane label, so discovery alone would return a subset and silently shrink the
affinity set -- losing capacity while looking healthy. The static list is a floor;
discovery only ever adds. The reverse case is safe on its own: a pool deleted from the
cluster but left in topology.yaml never matches nodeAffinity `In [...]`, and
reconcile_quota.py zeroes its quota.

TODO: every pool is meant to carry BOTH labels eventually -- ci-tpu-pool (unique per slice)
and ci-tpu-lane (shared by the lane). The pools listed in topology.yaml were created before
ci-tpu-lane existed and have only the first, which is the sole reason the static floor is
still here. Once they are all backfilled (README's "Cluster infra" table records the value
each one needs), `pools:` can drop out of topology.yaml and this becomes discovery-only:
delete the union below and the `static` fallback, keeping only `topology` and `queue` there.
Do NOT do it while any live pool is unlabelled -- a partial backfill is exactly the silent
capacity loss the union exists to prevent, so verify with
`kubectl get nodes -L ci-tpu-pool,ci-tpu-lane` that no CI pool has a blank lane first.

--sync also creates the Kueue ResourceFlavor for a newly discovered pool and lists it in
the lane's ClusterQueue -- the other half of what used to be a hand edit. Quota is a
starting value; reconcile_quota.py resets it from live chips before every submission.

Fails open to topology.yaml: no cluster, no kubectl, no RBAC -> today's behaviour. stdout
carries ONLY the space-separated pool list (the workflow captures it); everything else
goes to stderr.

Usage: discover_pools.py --lane nightly --hosts 2 [--sync] [--dry-run]
"""

import argparse
import json
import subprocess
import sys

# nodes_per_slice/check_single_slice live there; sys.path[0] is this dir when run by path.
import reconcile_quota
import yaml

POOL_LABEL = "ci-tpu-pool"
LANE_LABEL = "ci-tpu-lane"
TOPOLOGY_LABEL = "cloud.google.com/gke-tpu-topology"
TPU = "google.com/tpu"


def log(msg):
    print(msg, file=sys.stderr)


def kget(kind):
    out = subprocess.run(
        ["kubectl", "get", kind, "-o", "json"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout
    return json.loads(out)["items"]


def discover(nodes, lane, hosts):
    """{pool: topology} for pools in `lane` whose slice spans exactly `hosts` nodes."""
    found = {}
    for n in nodes:
        labels = n["metadata"].get("labels", {})
        pool = labels.get(POOL_LABEL)
        if not pool or labels.get(LANE_LABEL) != lane:
            continue
        if reconcile_quota.nodes_per_slice(n) != hosts:
            continue
        found[pool] = labels.get(TOPOLOGY_LABEL, "")
    return found


def chips_in_pool(nodes, pool):
    return sum(
        int(n["status"]["allocatable"].get(TPU, 0))
        for n in nodes
        if n["metadata"].get("labels", {}).get(POOL_LABEL) == pool
        and reconcile_quota.ready(n)
    )


def apply(doc, dry_run):
    if dry_run:
        log(f"(dry-run) would apply {doc['kind']}/{doc['metadata']['name']}")
        return
    subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=json.dumps(doc),
        text=True,
        check=True,
        timeout=60,
        capture_output=True,
    )


def ensure_flavor(pool, topology, existing, dry_run):
    """ResourceFlavor selecting `pool`, created if absent. Returns its name."""
    if pool in existing:
        return existing[pool]
    # Legacy flavor names are not derivable from the pool name (tpu7x-2x2x1-nightly selects
    # nightly-1h), which is why existing ones are looked up rather than recomputed. This
    # form is only ever minted for a pool that has no flavor, so it cannot collide.
    name = f"tpu7x-{topology or 'unknown'}-{pool}"
    log(f"creating ResourceFlavor {name} for pool {pool}")
    apply(
        {
            "apiVersion": "kueue.x-k8s.io/v1beta2",
            "kind": "ResourceFlavor",
            "metadata": {"name": name},
            "spec": {
                "nodeLabels": {POOL_LABEL: pool},
                "tolerations": [
                    {"key": TPU, "operator": "Exists", "effect": "NoSchedule"}
                ],
            },
        },
        dry_run,
    )
    return name


def ensure_in_queue(queue, flavor, chips, dry_run):
    """List `flavor` under `queue`'s TPU resource group if it is not there already."""
    cq = next(
        (c for c in kget("clusterqueues") if c["metadata"]["name"] == queue), None
    )
    if cq is None:
        log(f"⚠️  ClusterQueue {queue} not found; flavor {flavor} left unattached")
        return
    groups, changed = [], False
    for g in cq["spec"].get("resourceGroups", []):
        flavors = list(g.get("flavors", []))
        covered = g.get("coveredResources", [])
        if TPU in covered and not any(f["name"] == flavor for f in flavors):
            flavors.append(
                {
                    "name": flavor,
                    "resources": [
                        {"name": TPU, "nominalQuota": str(chips)},
                        {"name": "cpu", "nominalQuota": str(chips)},
                        {"name": "memory", "nominalQuota": f"{chips * 8}Gi"},
                    ],
                }
            )
            changed = True
        groups.append({"coveredResources": covered, "flavors": flavors})
    if not changed:
        return
    log(f"adding flavor {flavor} to ClusterQueue {queue} ({chips} chips)")
    if dry_run:
        log(f"(dry-run) would patch clusterqueue {queue}")
        return
    subprocess.run(
        [
            "kubectl",
            "patch",
            "clusterqueue",
            queue,
            "--type=merge",
            "-p",
            json.dumps({"spec": {"resourceGroups": groups}}),
        ],
        check=True,
        timeout=60,
        capture_output=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True)
    ap.add_argument("--hosts", type=int, required=True)
    ap.add_argument("--topology-file", default=".github/tpu-ci/topology.yaml")
    ap.add_argument(
        "--sync",
        action="store_true",
        help="create missing ResourceFlavors and attach them to the lane's ClusterQueue",
    )
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    entry = (
        yaml.safe_load(open(a.topology_file))["lanes"].get(a.lane, {}).get(a.hosts)
        or {}
    )
    static = list(entry.get("pools", []))
    if not static:
        log(f"⚠️  no topology.yaml entry for lane={a.lane} hosts={a.hosts}")

    try:
        nodes = kget("nodes")
        if not nodes:
            raise RuntimeError("no nodes returned")
        found = discover(nodes, a.lane, a.hosts)
    except Exception as e:  # noqa: BLE001 -- discovery must never block a test
        log(
            f"⚠️  pool discovery skipped ({type(e).__name__}: {e}); using topology.yaml"
        )
        print(" ".join(static))
        return

    # A multi-host pool holding two slices would let a gang straddle them and hang at ICI
    # rendezvous; refuse before that pool reaches a nodeAffinity list.
    for pool in found:
        reconcile_quota.check_single_slice(nodes, pool)

    new = [p for p in sorted(found) if p not in static]
    if new:
        log(f"discovered pools not in topology.yaml: {' '.join(new)}")
    if new and a.sync:
        # Registering a flavor needs cluster-scoped create on resourceflavors, which the CI
        # SA may not have. Swallow the failure: an unregistered pool costs one slice of
        # headroom, a raise here costs the whole test. Loud, because nothing else reports it.
        try:
            existing = {
                f["spec"]["nodeLabels"][POOL_LABEL]: f["metadata"]["name"]
                for f in kget("resourceflavors")
                if (f.get("spec") or {}).get("nodeLabels", {}).get(POOL_LABEL)
            }
            queue = entry.get("queue")
            for pool in new:
                flavor = ensure_flavor(pool, found[pool], existing, a.dry_run)
                if queue:
                    ensure_in_queue(
                        queue, flavor, chips_in_pool(nodes, pool), a.dry_run
                    )
        except Exception as e:  # noqa: BLE001 -- see above
            detail = getattr(e, "stderr", "") or ""
            log(
                f"⚠️  could not register {' '.join(new)} with Kueue "
                f"({type(e).__name__}: {e}) {detail.strip()[:300]}"
            )
            log("⚠️  Kueue will not admit onto those pools until this is fixed.")

    # Union collapses to `found` once every pool carries ci-tpu-lane -- see the TODO above.
    print(" ".join(sorted(set(static) | set(found))))


if __name__ == "__main__":
    main()
