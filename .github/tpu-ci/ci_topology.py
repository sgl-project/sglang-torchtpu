"""CI lanes and slice shapes from topology.yaml, and the Kueue names derived from them."""

import os

TOPOLOGY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "topology.yaml"
)


def load(path=TOPOLOGY_FILE):
    # Imported here so the split logic imports without site-packages (lint runs with -S).
    import yaml

    with open(path) as f:
        topo = yaml.safe_load(f)
    if topo["priority"] not in topo["lanes"]:
        raise ValueError(f"priority lane {topo['priority']!r} is not in lanes")
    return topo


def queue_name(lane, hosts):
    """ClusterQueue and LocalQueue name of `lane` on the shape with `hosts` hosts."""
    return f"tpu-{lane.replace('-', '')}-{hosts}h"


def cohort_name(shape):
    return f"ci-{shape}"


def shape_for_hosts(topo, hosts):
    """Topology label of the shape whose slice spans `hosts` hosts, or None."""
    return next(
        (name for name, shape in topo["shapes"].items() if shape["hosts"] == hosts),
        None,
    )
