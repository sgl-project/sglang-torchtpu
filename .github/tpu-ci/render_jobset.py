#!/usr/bin/env python3
"""Render the unified JobSet (jobset-tpu.yaml). Single source of truth for rendering --
used by _tpu-test.yml and runnable locally for dry-runs.

Two modes:
  single-host suite:  --suite <name>
      hosts=1; run_suite runs the suite (each test self-launches its own server).
  multi-host group:   --launch-script <rel-to-benchmark/tpu> --files "<f1> <f2> ..."
                      --hosts N [--weights-pvc <pvc>] [--name <run-name>]
      every host runs the launch script (it owns model+config); rank-0 runs each file.

The SLICE (topology + pool) is looked up by host count in topology.yaml. Only the render
vars are substituted; the container's own $spid / ${TPU_WORKER_ID} stay literal.
${WEIGHTS_VOLUME}/${WEIGHTS_MOUNT} expand to indented YAML blocks (empty when no PVC).
"""
import argparse
import re
import sys

import yaml

# Indentation matches jobset-tpu.yaml: volume items under `volumes:` at 16 spaces,
# volumeMount items under `volumeMounts:` at 20 spaces.
WEIGHTS_VOLUME_TMPL = (
    "                - name: weights\n"
    "                  persistentVolumeClaim:\n"
    "                    claimName: {pvc}\n"
    "                    readOnly: true"
)
WEIGHTS_MOUNT_TMPL = (
    "                    - name: weights\n"
    "                      mountPath: /weights\n"
    "                      readOnly: true"
)


def dns_safe(name):
    s = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    return (s[:40].strip("-")) or "run"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--nightly", action="store_true")
    ap.add_argument("--suite", default="")            # single-host mode
    ap.add_argument("--launch-script", default="")    # multi-host mode
    ap.add_argument("--files", default="")            # multi-host: space-separated
    ap.add_argument("--hosts", type=int, default=1)
    ap.add_argument("--weights-pvc", default="")
    ap.add_argument("--name", default="")             # multi-host run label
    ap.add_argument("--dir", default=".github/tpu-ci")
    a = ap.parse_args()

    slices = yaml.safe_load(open(f"{a.dir}/topology.yaml"))["slices"]

    if a.launch_script:
        hosts = a.hosts
        run_name = dns_safe(a.name or a.launch_script)
        suite = ""
        if not a.files:
            sys.exit("multi-host mode needs --files")
    else:
        if not a.suite:
            sys.exit("need --suite (single-host) or --launch-script (multi-host)")
        hosts = 1
        run_name = dns_safe(a.suite)
        suite = a.suite

    sl = slices.get(hosts, slices.get(str(hosts)))
    if not sl:
        sys.exit(f"no slice for hosts={hosts} in topology.yaml")

    pvc = a.weights_pvc
    vals = {
        "IMAGE": a.image,
        "HOSTS": str(hosts),
        "TOPOLOGY": sl["topology"],
        "POOL": sl["pool"],
        "RUN_NAME": run_name,
        "SUITE": suite,
        "NIGHTLY_FLAG": "--nightly" if a.nightly else "",
        "LAUNCH_SCRIPT": a.launch_script,
        "TEST_FILES": a.files,
        "WEIGHTS_VOLUME": WEIGHTS_VOLUME_TMPL.format(pvc=pvc) if pvc else "",
        "WEIGHTS_MOUNT": WEIGHTS_MOUNT_TMPL if pvc else "",
    }

    tpl = open(f"{a.dir}/jobset-tpu.yaml").read()
    out = re.sub(r"\$\{(\w+)\}", lambda m: vals.get(m.group(1), m.group(0)), tpl)
    sys.stdout.write(out)


if __name__ == "__main__":
    main()
