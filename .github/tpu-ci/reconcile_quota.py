#!/usr/bin/env python3
"""Set each CI ClusterQueue's nominalQuota from live allocatable TPU chips.

Run by _tpu-test.yml right before it submits a JobSet, so quota reflects the cluster at
the moment admission is decided. Between submissions a stale quota affects nothing --
nothing is consuming it -- and waitForPodsReady covers capacity lost while a job is
already queued or running.

A flavor is skipped unless it selects on ci-tpu-pool, so non-CI queues are untouched.
cpu and memory are derived from the chip count (pods take 4 chips / 4 cpu / 32Gi, so
1 cpu and 8Gi per chip) to keep TPU the only binding resource.

Fails open: a reconcile that cannot run leaves the existing quota in place and warns,
rather than blocking the test.

Usage: reconcile_quota.py [--dry-run] [--pool=<ci-tpu-pool>]
"""

import json
import re
import subprocess
import sys

POOL_LABEL = "ci-tpu-pool"
TPU = "google.com/tpu"


def kget(kind):
    out = subprocess.run(
        ["kubectl", "get", kind, "-o", "json"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout
    return json.loads(out)["items"]


def ready(node):
    return any(
        c["type"] == "Ready" and c["status"] == "True"
        for c in node["status"].get("conditions", [])
    )


def nodes_per_slice(node):
    """How many nodes one slice of this node's topology spans.

    2x2x1 = 4 chips = 1 node (each node its own slice); 2x2x2 = 8 = 2; 2x2x4 = 16 = 4.
    """
    labels = node["metadata"]["labels"]
    dims = labels.get("cloud.google.com/gke-tpu-topology", "")
    if not re.fullmatch(r"\d+(x\d+)*", dims):
        return 1
    chips = 1
    for d in dims.split("x"):
        chips *= int(d)
    per_node = int(labels.get("cloud.google.com/gke-accelerator-count", 4)) or 4
    return max(1, chips // per_node)


def check_single_slice(nodes, pool):
    """A multi-host pool must hold exactly one slice.

    GKE tpu7x nodes carry no per-slice label -- ci-tpu-pool is the finest selector there
    is -- so two slices behind one label lets a JobSet's pods straddle them. They start,
    find no ICI path to each other, and hang until the deadline with no error saying why.
    Resizing a multi-host pool is the way this happens.
    """
    members = [
        n for n in nodes if n["metadata"].get("labels", {}).get(POOL_LABEL) == pool
    ]
    if not members:
        return
    want = nodes_per_slice(members[0])
    if want > 1 and len(members) != want:
        topo = members[0]["metadata"]["labels"].get("cloud.google.com/gke-tpu-topology")
        sys.exit(
            f"pool {pool!r} has {len(members)} nodes but one {topo} slice is {want}. "
            "Pods could straddle two slices and hang at ICI rendezvous -- add capacity as "
            "a NEW pool with its own ci-tpu-pool label, never by resizing this one."
        )


def chips_per_pool(nodes):
    chips = {}
    for n in nodes:
        pool = n["metadata"].get("labels", {}).get(POOL_LABEL)
        if pool and ready(n):
            alloc = int(n["status"]["allocatable"].get(TPU, 0))
            chips[pool] = chips.get(pool, 0) + alloc
    return chips


def main():
    dry_run = "--dry-run" in sys.argv
    pool = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--pool=")), "")

    nodes = kget("nodes")
    if not nodes:
        sys.exit("no nodes returned -- refusing to zero every quota")
    if pool:
        check_single_slice(nodes, pool)
    chips = chips_per_pool(nodes)

    flavor_pool = {
        f["metadata"]["name"]: f["spec"]["nodeLabels"][POOL_LABEL]
        for f in kget("resourceflavors")
        if (f.get("spec") or {}).get("nodeLabels", {}).get(POOL_LABEL)
    }

    for cq in kget("clusterqueues"):
        name = cq["metadata"]["name"]
        groups, changed = [], False
        for g in cq["spec"].get("resourceGroups", []):
            flavors = []
            for fl in g.get("flavors", []):
                pool = flavor_pool.get(fl["name"])
                if pool is None:
                    flavors.append(fl)
                    continue
                n = chips.get(pool, 0)
                want = {TPU: str(n), "cpu": str(n), "memory": f"{n * 8}Gi"}
                have = {r["name"]: str(r["nominalQuota"]) for r in fl["resources"]}
                if {k: have.get(k) for k in want} != want:
                    changed = True
                    print(f"{name}/{fl['name']} ({pool}): {have} -> {want}")
                flavors.append(
                    {
                        "name": fl["name"],
                        "resources": [
                            {"name": k, "nominalQuota": v} for k, v in want.items()
                        ],
                    }
                )
            groups.append(
                {"coveredResources": g["coveredResources"], "flavors": flavors}
            )
        if not changed:
            continue
        if dry_run:
            print(f"(dry-run) would patch {name}")
            continue
        subprocess.run(
            [
                "kubectl",
                "patch",
                "clusterqueue",
                name,
                "--type=merge",
                "-p",
                json.dumps({"spec": {"resourceGroups": groups}}),
            ],
            check=True,
            timeout=60,
        )
        print(f"patched {name}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 -- stale quota must not block the test
        print(
            f"⚠️  quota reconcile skipped ({type(e).__name__}: {e}); "
            "proceeding with existing quota"
        )
