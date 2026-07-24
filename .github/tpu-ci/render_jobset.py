#!/usr/bin/env python3
"""Render the single-host JobSet (jobset-tpu.yaml) for a suite.

Single source of truth for rendering -- used by _tpu-test.yml and runnable locally for
dry-runs. Looks up the slice (topology + pool) for the host count in topology.yaml and
substitutes the render vars; the container's own $VAR stays literal.

Usage: render_jobset.py --image <ref> --suite <name> [--nightly] > job.yaml
"""
import argparse
import re
import sys

import yaml


def dns_safe(name):
    s = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    return (s[:40].strip("-")) or "run"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--nightly", action="store_true")
    ap.add_argument("--dir", default=".github/tpu-ci")
    a = ap.parse_args()

    slices = yaml.safe_load(open(f"{a.dir}/topology.yaml"))["slices"]
    hosts = 1
    sl = slices.get(hosts, slices.get(str(hosts)))
    if not sl:
        sys.exit(f"no slice for hosts={hosts} in topology.yaml")

    vals = {
        "IMAGE": a.image,
        "HOSTS": str(hosts),
        "TOPOLOGY": sl["topology"],
        "POOL": sl["pool"],
        "RUN_NAME": dns_safe(a.suite),
        "SUITE": a.suite,
        "NIGHTLY_FLAG": "--nightly" if a.nightly else "",
    }

    tpl = open(f"{a.dir}/jobset-tpu.yaml").read()
    out = re.sub(r"\$\{(\w+)\}", lambda m: vals.get(m.group(1), m.group(0)), tpl)
    sys.stdout.write(out)


if __name__ == "__main__":
    main()
