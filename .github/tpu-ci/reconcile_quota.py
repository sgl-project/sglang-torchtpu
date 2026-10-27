#!/usr/bin/env python3
"""Keep the CI's Kueue objects in step with topology.yaml and the live CI node pools.

For each shape in topology.yaml this ensures:
  - one ResourceFlavor per CI pool of the shape (a flavor is one pool, so a multi-host gang
    lands on one slice);
  - one ClusterQueue + LocalQueue per lane, all in the shape's cohort, each listing every
    pool of the shape;
  - quotas from live Ready nodes: each lane's min_hosts in whole slices, never lent, and
    the priority lane holding the rest, lent while it is idle.

Run by _tpu-test.yml before every submission, so quota matches the cluster when admission
is decided. Any error fails the job: a stale quota admits jobs onto chips that do not
exist, or leaves live chips idle, and nothing downstream reports either.

cpu and memory are derived from the chip count (pods take 4 chips / 4 cpu / 32Gi, so 1 cpu
and 8Gi per chip) to keep TPU the only binding resource.

Usage: reconcile_quota.py [--dry-run]
"""

import json
import re
import subprocess
import sys

import ci_topology

POOL_LABEL = "ci-tpu-pool"
TOPOLOGY_LABEL = "cloud.google.com/gke-tpu-topology"
TPU = "google.com/tpu"
NAMESPACE = "tpu-ci"
KUEUE_API = "kueue.x-k8s.io/v1beta2"


def kget(kind, namespace=None):
    cmd = ["kubectl", "get", kind, "-o", "json"]
    if namespace:
        cmd += ["-n", namespace]
    out = subprocess.run(
        cmd, capture_output=True, text=True, timeout=60, check=True
    ).stdout
    return json.loads(out)["items"]


def ready(node):
    return any(
        c["type"] == "Ready" and c["status"] == "True"
        for c in node["status"].get("conditions", [])
    )


def nodes_per_slice(node):
    """2x2x1 = 4 chips = 1 node; 2x2x2 = 8 = 2; 2x2x4 = 16 = 4."""
    labels = node["metadata"]["labels"]
    dims = labels.get(TOPOLOGY_LABEL, "")
    if not re.fullmatch(r"\d+(x\d+)*", dims):
        return 1
    chips = 1
    for d in dims.split("x"):
        chips *= int(d)
    return max(1, chips // chips_per_host(node))


def chips_per_host(node):
    labels = node["metadata"]["labels"]
    return int(labels.get("cloud.google.com/gke-accelerator-count", 4)) or 4


def ci_pools(nodes):
    """{pool: {topology, hosts, nodes, ready, chips_per_host}} for every CI pool."""
    pools = {}
    for n in nodes:
        labels = n["metadata"].get("labels", {})
        pool = labels.get(POOL_LABEL)
        if not pool:
            continue
        p = pools.setdefault(
            pool,
            {
                "topology": labels.get(TOPOLOGY_LABEL, ""),
                "hosts": nodes_per_slice(n),
                "nodes": 0,
                "ready": 0,
                "chips_per_host": chips_per_host(n),
            },
        )
        p["nodes"] += 1
        p["ready"] += ready(n)
    return pools


def usable_slices(pool, info):
    """Whole slices `pool` can host right now.

    GKE tpu7x nodes carry no per-slice label, so a multi-host pool holding more than one
    slice would let a gang straddle two of them and hang at ICI rendezvous. Such a pool
    gets no quota; add capacity as a new pool, never by resizing a multi-host one.
    """
    if info["hosts"] == 1:
        return info["ready"]
    if info["nodes"] != info["hosts"]:
        print(
            f"::warning::pool {pool!r} has {info['nodes']} nodes but one "
            f"{info['topology']} slice is {info['hosts']}; giving it no quota",
            file=sys.stderr,
        )
        return 0
    return int(info["ready"] == info["hosts"])


def split_quota(slices, hosts, chips_per_slice, lanes, priority, min_hosts):
    """{lane: {pool: (chips, lendable_chips)}} for one shape.

    `slices` maps each pool of the shape to its usable slices. Lanes take their min_hosts
    first, in `lanes` order, from the pools with the most slices first (a reserve spans as
    few pools as possible); the priority lane gets the rest and may lend all of it.
    """
    left = dict(sorted(slices.items(), key=lambda kv: (-kv[1], kv[0])))
    out = {lane: {pool: [0, 0] for pool in left} for lane in lanes}
    for lane in lanes:
        want = min_hosts.get(lane, 0)
        if want % hosts:
            sys.exit(
                f"min_hosts {want} for {lane} is not a whole number of {hosts}-host slices"
            )
        need = want // hosts
        for pool in left:
            take = min(need, left[pool])
            left[pool] -= take
            need -= take
            out[lane][pool][0] += take * chips_per_slice
    for pool, n in left.items():
        out[priority][pool][0] += n * chips_per_slice
        out[priority][pool][1] += n * chips_per_slice
    return {
        lane: {p: tuple(v) for p, v in by_pool.items()} for lane, by_pool in out.items()
    }


def quantity(n, unit=""):
    # The API server stores a zero quantity as "0", whatever unit it was written with.
    return f"{n}{unit}" if n else "0"


def flavor_resources(chips, lendable):
    return [
        {
            "name": TPU,
            "nominalQuota": quantity(chips),
            "lendingLimit": quantity(lendable),
        },
        {"name": "cpu", "nominalQuota": quantity(chips)},
        {"name": "memory", "nominalQuota": quantity(chips * 8, "Gi")},
    ]


def cluster_queue_spec(shape, flavors):
    """`flavors`: [(flavor name, chips, lendable chips)] in the order Kueue tries them."""
    return {
        "namespaceSelector": {},
        "queueingStrategy": "BestEffortFIFO",
        "cohortName": ci_topology.cohort_name(shape),
        # Use a pool where the job fits in this queue's own quota before borrowing.
        "flavorFungibility": {"whenCanBorrow": "TryNextFlavor"},
        # Lent slices come back as borrowed jobs finish.
        "preemption": {"reclaimWithinCohort": "Never"},
        "resourceGroups": [
            {
                "coveredResources": [TPU, "cpu", "memory"],
                "flavors": [
                    {
                        "name": name,
                        "resources": flavor_resources(chips=chips, lendable=lendable),
                    }
                    for name, chips, lendable in flavors
                ],
            }
        ],
    }


def normalized(spec):
    """The fields cluster_queue_spec sets, with quantities as strings, for comparison."""
    groups = [
        {
            "coveredResources": g.get("coveredResources"),
            "flavors": [
                {
                    "name": f["name"],
                    "resources": [
                        {k: str(v) for k, v in r.items()}
                        for r in f.get("resources", [])
                    ],
                }
                for f in g.get("flavors", [])
            ],
        }
        for g in spec.get("resourceGroups", [])
    ]
    return {
        "cohortName": spec.get("cohortName"),
        "whenCanBorrow": (spec.get("flavorFungibility") or {}).get("whenCanBorrow"),
        "reclaimWithinCohort": (spec.get("preemption") or {}).get(
            "reclaimWithinCohort"
        ),
        "resourceGroups": groups,
    }


def kubectl(args, doc, dry_run):
    """Run a write; --dry-run still sends it, so the API server and Kueue validate it.

    Objects are created with `apply`: CI jobs start together, and two creating the same
    missing queue must both succeed.
    """
    if dry_run:
        args = [*args, "--dry-run=server"]
    print(f"kubectl {' '.join(args)}")
    subprocess.run(
        ["kubectl", *args],
        input=json.dumps(doc) if doc is not None else None,
        text=True,
        check=True,
        timeout=60,
    )


def ensure_flavor(pool, shape, existing, dry_run):
    """Name of the ResourceFlavor selecting `pool`, created if absent."""
    if pool in existing:
        return existing[pool]
    name = f"tpu7x-{shape}-{pool}"
    doc = {
        "apiVersion": KUEUE_API,
        "kind": "ResourceFlavor",
        "metadata": {"name": name},
        "spec": {
            "nodeLabels": {POOL_LABEL: pool},
            "tolerations": [{"key": TPU, "operator": "Exists", "effect": "NoSchedule"}],
        },
    }
    kubectl(args=["apply", "-f", "-"], doc=doc, dry_run=dry_run)
    return name


def ensure_cluster_queue(name, spec, live, dry_run):
    if live is None:
        doc = {
            "apiVersion": KUEUE_API,
            "kind": "ClusterQueue",
            "metadata": {"name": name},
            "spec": spec,
        }
        kubectl(args=["apply", "-f", "-"], doc=doc, dry_run=dry_run)
    elif normalized(live["spec"]) != normalized(spec):
        # A merge patch replaces each list whole, so flavors of deleted pools drop out.
        patch = json.dumps({"spec": spec})
        kubectl(
            args=["patch", "clusterqueue", name, "--type=merge", "-p", patch],
            doc=None,
            dry_run=dry_run,
        )


def ensure_local_queue(name, existing, dry_run):
    if name in existing:
        return
    doc = {
        "apiVersion": KUEUE_API,
        "kind": "LocalQueue",
        "metadata": {"name": name, "namespace": NAMESPACE},
        "spec": {"clusterQueue": name},
    }
    kubectl(args=["apply", "-f", "-"], doc=doc, dry_run=dry_run)


def sync_shape(shape, cfg, pools, topo, live, dry_run):
    hosts = cfg["hosts"]
    members = sorted(p for p, info in pools.items() if info["topology"] == shape)
    if not members:
        print(
            f"::warning::no CI pool of shape {shape}; its queues are left as they are"
        )
        return
    per_host = pools[members[0]]["chips_per_host"]
    slices = {p: usable_slices(pool=p, info=pools[p]) for p in members}
    # Queues try flavors in this order too, so a lane looks in its reserve's pools first.
    members.sort(key=lambda p: (-slices[p], p))
    split = split_quota(
        slices=slices,
        hosts=hosts,
        chips_per_slice=hosts * per_host,
        lanes=topo["lanes"],
        priority=topo["priority"],
        min_hosts=cfg.get("min_hosts", {}),
    )
    flavor = {
        p: ensure_flavor(pool=p, shape=shape, existing=live["flavors"], dry_run=dry_run)
        for p in members
    }
    for lane in topo["lanes"]:
        name = ci_topology.queue_name(lane, hosts)
        quotas = [(flavor[p], *split[lane][p]) for p in members]
        print(
            f"{name}: "
            + ", ".join(f"{p}={c} chips, lend {l}" for p, (c, l) in split[lane].items())
        )
        ensure_cluster_queue(
            name=name,
            spec=cluster_queue_spec(shape=shape, flavors=quotas),
            live=live["cqs"].get(name),
            dry_run=dry_run,
        )
        ensure_local_queue(name=name, existing=live["lqs"], dry_run=dry_run)


def main():
    dry_run = "--dry-run" in sys.argv
    topo = ci_topology.load()
    nodes = kget("nodes")
    if not nodes:
        sys.exit("no nodes returned -- refusing to zero every quota")
    pools = ci_pools(nodes)
    live = {
        "flavors": {
            f["spec"]["nodeLabels"][POOL_LABEL]: f["metadata"]["name"]
            for f in kget("resourceflavors")
            if (f.get("spec") or {}).get("nodeLabels", {}).get(POOL_LABEL)
        },
        "cqs": {c["metadata"]["name"]: c for c in kget("clusterqueues")},
        "lqs": {
            q["metadata"]["name"] for q in kget(kind="localqueues", namespace=NAMESPACE)
        },
    }
    for shape, cfg in topo["shapes"].items():
        sync_shape(
            shape=shape, cfg=cfg, pools=pools, topo=topo, live=live, dry_run=dry_run
        )
    unlisted = sorted({i["topology"] for i in pools.values()} - set(topo["shapes"]))
    if unlisted:
        print(
            f"::warning::CI pools of shapes not in topology.yaml get no queue: {unlisted}"
        )


if __name__ == "__main__":
    main()
