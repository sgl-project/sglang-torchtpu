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

Also computes the CI poll budget for a suite (--print-wait-seconds): a static AST scan
of the registered tree sums est_time over the suite's enabled tests -- no sglang import
needed (same technique as discover_multihost.py). The workflow uses it so the poll
timeout scales with the suite instead of a fixed ceiling.

Usage: render_jobset.py --image <ref> --suite <name> [--nightly] > job.yaml
       render_jobset.py --image <ref> --launch-script <rel> --files "<f>..." --hosts N
       render_jobset.py --suite <name> --print-wait-seconds
"""

import argparse
import ast
import os
import re
import sys

import yaml

# Poll budget = max(sum(est_time) * MARGIN, FLOOR). The margin gives headroom over the
# (approximate) est_time estimates; the floor covers tiny/near-empty suites.
_WAIT_MARGIN = 1.5
_WAIT_FLOOR_SECONDS = 360

# Positional order of register_*_ci(...) args (mirrors ci_register._PARAM_ORDER) so the
# scan handles both register_tpu_ci(120, "suite") and register_tpu_ci(est_time=120, ...).
_REGISTER_PARAMS = ("est_time", "suite", "nightly", "disabled")

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


def suite_wait_seconds(suite, tests_dir="test/registered"):
    """CI poll budget (seconds) for `suite`: max(sum(est_time) * MARGIN, FLOOR).

    Sums est_time over the enabled tests registered to `suite` by statically parsing
    the registered tree (no sglang import). A test counts if it has a register_*_ci(...)
    call with suite=<suite> and no truthy `disabled=` reason.
    """
    total = 0.0
    for root, _dirs, files in os.walk(tests_dir):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            try:
                tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fname = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
                if not (fname.startswith("register_") and fname.endswith("_ci")):
                    continue
                # Merge positional + keyword args by name so both call styles work.
                argv = {
                    _REGISTER_PARAMS[i]: a
                    for i, a in enumerate(node.args)
                    if i < len(_REGISTER_PARAMS)
                }
                argv.update({k.arg: k.value for k in node.keywords if k.arg})
                suite_val = argv.get("suite")
                if not (
                    isinstance(suite_val, ast.Constant) and suite_val.value == suite
                ):
                    continue
                disabled = argv.get("disabled")
                if disabled is not None and not (
                    isinstance(disabled, ast.Constant) and not disabled.value
                ):
                    continue  # disabled registration -- not run, so excluded
                est = argv.get("est_time")
                if isinstance(est, ast.Constant) and isinstance(
                    est.value, (int, float)
                ):
                    total += float(est.value)
    return max(int(total * _WAIT_MARGIN), _WAIT_FLOOR_SECONDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--nightly", action="store_true")
    ap.add_argument("--suite", default="")  # single-host mode
    ap.add_argument("--launch-script", default="")  # multi-host mode
    ap.add_argument("--files", default="")  # multi-host: space-separated
    # Exec mode: every host runs this command; pod exit codes gate the JobSet.
    # Plain words only -- the command is inlined into the pod script unquoted.
    ap.add_argument("--exec", dest="exec_cmd", default="")
    ap.add_argument("--hosts", type=int, default=1)
    ap.add_argument("--weights-pvc", default="")
    ap.add_argument("--name", default="")  # multi-host run label
    ap.add_argument("--dir", default=".github/tpu-ci")
    ap.add_argument("--tests-dir", default="test/registered")
    ap.add_argument(
        "--print-wait-seconds",
        action="store_true",
        help="Print the CI poll budget (s) for --suite and exit (no rendering).",
    )
    a = ap.parse_args()

    if a.print_wait_seconds:
        print(suite_wait_seconds(a.suite, a.tests_dir))
        return

    if not a.image:
        ap.error("--image is required when rendering")

    slices = yaml.safe_load(open(f"{a.dir}/topology.yaml"))["slices"]

    if a.launch_script:
        if a.hosts < 1:
            ap.error("--hosts must be a positive integer for multi-host mode")
        hosts = a.hosts
        run_name = dns_safe(a.name or a.launch_script)
        suite = ""
        if not a.files:
            sys.exit("multi-host mode needs --files")
    elif a.exec_cmd:
        if a.hosts < 1:
            ap.error("--hosts must be a positive integer for exec mode")
        hosts = a.hosts
        run_name = dns_safe(a.name or "exec")
        suite = ""
    else:
        if not a.suite:
            sys.exit(
                "need --suite (single-host), --launch-script (multi-host) or --exec"
            )
        hosts = 1
        run_name = dns_safe(a.suite)
        suite = a.suite

    sl = slices.get(hosts, slices.get(str(hosts)))
    if not sl:
        sys.exit(f"no slice for hosts={hosts} in topology.yaml")

    # Suite mode runs every registered file sequentially, so its deadline scales
    # with the suite (same est_time formula as _tpu-test's poll budget, keeping the
    # two in agreement). Exec gets 7200: the combo rows budget WAIT_SECS=2400 for
    # engine bring-up alone.
    if suite:
        active_deadline = suite_wait_seconds(suite, a.tests_dir)
    elif a.exec_cmd:
        active_deadline = 7200
    else:
        active_deadline = 5400

    pvc = a.weights_pvc
    vals = {
        "IMAGE": a.image,
        "ACTIVE_DEADLINE": str(active_deadline),
        "EXEC_CMD": a.exec_cmd,
        "HOSTS": str(hosts),
        "TOPOLOGY": sl["topology"],
        "POOL": sl["pool"],
        "QUEUE": sl["queue"],
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
