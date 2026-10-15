"""Print each cluster's live TPU slices per nodepool, and how many hosts are busy.

The docs and Kueue quotas describe intended capacity; this reads the nodes themselves.

Usage:
  tpu_inventory.py --context <name>=<kube context> [--context ...] [--summary <file>]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import defaultdict

CHIPS_PER_HOST = 4  # tpu7x-standard-4t
NODEPOOL = "cloud.google.com/gke-nodepool"
TOPOLOGY = "cloud.google.com/gke-tpu-topology"


def kubectl_json(context: str, *args: str) -> dict:
    out = subprocess.run(
        ["kubectl", f"--context={context}", *args, "-o", "json"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out)


def busy_nodes(context: str) -> set[str]:
    """Nodes holding a pod that requests TPU chips and has not finished."""
    busy = set()
    for pod in kubectl_json(context, "get", "pods", "-A")["items"]:
        node = pod["spec"].get("nodeName")
        if not node or pod["status"].get("phase") in ("Succeeded", "Failed"):
            continue
        for c in pod["spec"]["containers"]:
            if "google.com/tpu" in (c.get("resources", {}).get("requests") or {}):
                busy.add(node)
    return busy


def inventory(name: str, context: str) -> list[str]:
    nodes = kubectl_json(
        context, "get", "nodes", "-l", "cloud.google.com/gke-tpu-accelerator"
    )["items"]
    try:
        busy = busy_nodes(context)
    except subprocess.CalledProcessError as e:
        print(
            f"[{name}] cannot list pods cluster-wide ({e.stderr.strip()}); busy column unknown"
        )
        busy = None
    pools: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for n in nodes:
        lb = n["metadata"]["labels"]
        key = (
            lb.get(NODEPOOL, "?"),
            lb.get(TOPOLOGY, "?"),
            lb.get("ci-tpu-pool", "-"),
            lb.get("ci-tpu-lane", "-"),
        )
        pools[key].append(n["metadata"]["name"])
    rows = [
        f"#### {name}",
        "",
        "| nodepool | topology | ci-tpu-pool | ci-tpu-lane | hosts | slices | chips | busy hosts |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    tot_hosts = tot_slices = 0
    for (pool, topo, ci_pool, lane), names in sorted(pools.items()):
        dims = [int(d) for d in topo.split("x")] if topo != "?" else [CHIPS_PER_HOST]
        hosts_per_slice = max(1, math.prod(dims) // CHIPS_PER_HOST)
        slices = len(names) / hosts_per_slice
        nbusy = "?" if busy is None else str(sum(x in busy for x in names))
        rows.append(
            f"| {pool} | {topo} | {ci_pool} | {lane} | {len(names)} | {slices:g} "
            f"| {len(names) * CHIPS_PER_HOST} | {nbusy} |"
        )
        tot_hosts += len(names)
        tot_slices += slices
    rows.append(
        f"| **total** | | | | {tot_hosts} | {tot_slices:g} | {tot_hosts * CHIPS_PER_HOST} | |"
    )
    return rows + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", action="append", default=[], help="name=kube-context")
    ap.add_argument("--summary", default="")
    a = ap.parse_args()
    out = ["### TPU inventory", ""]
    for spec in a.context:
        name, _, ctx = spec.partition("=")
        if not ctx:
            continue
        try:
            out += inventory(name, ctx)
        except subprocess.CalledProcessError as e:
            out += [f"#### {name}", "", f"inventory failed: {e.stderr.strip()}", ""]
    text = "\n".join(out)
    print(text)
    if a.summary:
        with open(a.summary, "a") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
