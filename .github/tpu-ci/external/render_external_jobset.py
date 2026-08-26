"""Render the external multi-host JobSet (jobset-external-tpu.yaml), and discover the
`external-nightly-*` work for the nightly matrix. Single source of truth for rendering onto
the external TPU cluster -- used by _external-tpu-test.yml and runnable locally for dry-runs.
See external-topology.yaml + README.md.

REUSE: the registration scan (`discover_multihost._scan`, incl. its `disabled` handling) and
the path-derived naming (`test_names.names_from_path`, from #552) are imported from the
in-project CI helpers, so improvements to those land here too. What stays unique to the
external path lives in this file: slice bucketing (no Kueue), gcsfuse weights, the
nodepool-pin JobSet, and the DNS-endpoint gating (in _external-tpu-test.yml).

An external multi-host test declares a module-level MULTIHOST_LAUNCH dict and registers to an
external suite:

    MULTIHOST_LAUNCH = {
        "launch_script": "recipe/qwen3_coder_30b_a3b_fp8.sh",  # relative to benchmark/tpu/
        "weights": "Qwen3-Coder-30B-A3B-Instruct-FP8",         # subdir under gs://<bucket>/weights/
        # "extra_args": "...",   # optional; overrides weights.default_extra_args
    }
    register_tpu_ci(est_time=1200, suite="external-nightly-2x2x2", nightly=True)

The SUITE fixes the slice shape (hosts/topology/nodepools live in external-topology.yaml).
tp = hosts x devices_per_host.

Modes:
  --list-groups [--nightly]
      Scan test/registered/**.py (AST only), group files sharing (suite, launch_script,
      weights) into one launch group (they share one server), then -- because there is no
      Kueue to spread work -- ROUND-ROBIN each suite's groups across its slices and emit ONE
      matrix row PER SLICE. Each row carries the slice's ordered `groups`; the workflow runs
      them back-to-back on that one slice (parallel across slices, serial within a slice):
        [{"name","suite","hosts","topology","nodepool_var","groups":[{...}]}]
      `nodepool_var` is the NAME of the repo secret holding that slice's nodepool; the
      workflow resolves it so the value never appears in source or the matrix.
  render (default)
      Render ONE group's JobSet (the workflow calls this once per group in a slice):
      --image <ref> --suite <external-suite> --launch-script <rel> --files "<f>..." \
          --weights <model-subdir> --nodepool <value> [--name <run-name>] [--extra-args "<flags>"]

Usage: render_external_jobset.py --list-groups
       render_external_jobset.py --image <ref> --suite external-nightly-2x2x2 --launch-script recipe/x.sh \
           --files "a.py b.py" --weights <model-subdir> --nodepool <value> > job.yaml
"""

import argparse
import json
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

# Reuse the in-project CI helpers (external/ is a subdir, so put the parent tpu-ci dir on the
# path). _scan gives us registration + `disabled` handling; names_from_path gives the #552
# path-derived display/compact names. Importing here means future changes to them flow through.
sys.path.insert(0, os.path.dirname(HERE))
from discover_multihost import _scan as _scan_suite  # noqa: E402
from test_names import names_from_path  # noqa: E402

# ~30m bring-up (weight load over gcsfuse + compile warmup) + rank-0's test files + margin.
_ACTIVE_DEADLINE = 9000


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def dns_safe(name):
    s = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    return (s[:40].strip("-")) or "run"


def list_groups(cfg, root, nightly_only):
    suites = cfg["suites"]

    # 1. Scan per suite (reuse the in-project registration scan) -> {path: (suite, launch)}.
    scanned = {}
    for suite in suites:
        for path, launch in _scan_suite(root, suite, nightly_only).items():
            scanned[path] = (suite, launch)

    # 2. Group files that share a server: same (suite, launch_script, weights).
    groups = {}
    for path, (suite, spec) in sorted(scanned.items()):
        ls, weights = spec.get("launch_script"), spec.get("weights")
        if not ls or not weights:
            print(
                f"skip {path}: MULTIHOST_LAUNCH needs launch_script + weights, got {spec!r}",
                file=sys.stderr,
            )
            continue
        groups.setdefault((suite, ls, weights), []).append(path)

    # 3. Round-robin each suite's groups across its slices and BUCKET by slice, so the matrix
    #    is one job PER SLICE. Deterministic (sorted keys). The i-th group of a suite -> slice
    #    i % n; groups on the same slice run serially inside that one job (cross-group conc=1),
    #    while different slices run in parallel.
    slices = {}
    per_suite_i = {}
    for (suite, ls, weights), files in sorted(groups.items()):
        sl = suites[suite]
        pool = sl["nodepool_vars"]
        i = per_suite_i.get(suite, 0)
        per_suite_i[suite] = i + 1
        slot = i % len(pool)
        bucket = slices.setdefault(
            (suite, slot),
            {
                "name": f"{suite}-s{slot}",
                "suite": suite,
                "hosts": sl["hosts"],
                "topology": sl["topology"],
                "nodepool_var": pool[slot],
                "groups": [],
            },
        )
        bucket["groups"].append(
            {
                **names_from_path(
                    sorted(files)[0]
                ),  # #552: path-derived name + compact run_name
                "launch_script": ls,
                "weights": weights,
                "files": " ".join(sorted(files)),
            }
        )
    return [slices[k] for k in sorted(slices)]


def render(cfg, a):
    suites = cfg["suites"]
    if a.suite not in suites:
        sys.exit(
            f"unknown --suite {a.suite!r}; external-topology.yaml has {sorted(suites)}"
        )
    if not a.launch_script or not a.files or not a.weights:
        sys.exit("render needs --launch-script, --files and --weights")

    if not a.nodepool:
        sys.exit(
            "render needs --nodepool (the resolved slice nodepool, from a repo secret)"
        )

    sl = suites[a.suite]
    cl = cfg["cluster"]
    w = cfg["weights"]
    hosts = int(sl["hosts"])
    tp = hosts * int(cl["devices_per_host"])
    model_path = f"{w['mount_path']}/{w['prefix']}/{a.weights}"
    extra_args = a.extra_args or w.get("default_extra_args", "")

    vals = {
        "IMAGE": a.image,
        # --name is the group's compact run_name (names_from_path) -> collision-resistant.
        "RUN_NAME": dns_safe(a.name or a.suite),
        "NAMESPACE": cl["namespace"],
        "HOSTS": str(hosts),
        "TOPOLOGY": sl["topology"],
        "NODEPOOL": a.nodepool,
        "ACCELERATOR": cl["accelerator"],
        "ACTIVE_DEADLINE": str(a.active_deadline),
        "LAUNCH_SCRIPT": a.launch_script,
        "TEST_FILES": a.files,
        "MODEL_PATH": model_path,
        "TP_SIZE": str(tp),
        "EP_SIZE": str(tp),
        "EXTRA_ARGS": extra_args,
        "TIMEOUT_PER_FILE": str(a.timeout_per_file),
        "WEIGHTS_BUCKET": w["bucket"],
        "WEIGHTS_MOUNT_OPTIONS": w["mount_options"],
        "WEIGHTS_MOUNT_PATH": w["mount_path"],
        "EPHEMERAL_LIMIT": w["ephemeral_storage_limit"],
    }
    tpl = open(f"{HERE}/jobset-external-tpu.yaml").read()
    out = re.sub(r"\$\{(\w+)\}", lambda m: vals.get(m.group(1), m.group(0)), tpl)
    sys.stdout.write(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=f"{HERE}/external-topology.yaml")
    ap.add_argument("--tests-dir", default="test/registered")
    ap.add_argument(
        "--list-groups",
        action="store_true",
        help="Print the JSON per-slice matrix for the external suites and exit.",
    )
    ap.add_argument("--nightly", action="store_true", default=True)
    # render mode (one group)
    ap.add_argument("--image")
    ap.add_argument("--suite", default="")
    ap.add_argument("--launch-script", default="")
    ap.add_argument("--files", default="")
    ap.add_argument("--weights", default="")
    ap.add_argument(
        "--nodepool",
        default="",
        help="Resolved slice nodepool (the workflow reads it from the suite's repo secret).",
    )
    ap.add_argument("--name", default="")
    ap.add_argument("--extra-args", default="")
    ap.add_argument("--active-deadline", type=int, default=_ACTIVE_DEADLINE)
    ap.add_argument("--timeout-per-file", type=int, default=3600)
    a = ap.parse_args()

    cfg = load_config(a.config)

    if a.list_groups:
        print(json.dumps(list_groups(cfg, a.tests_dir, a.nightly)))
        return

    if not a.image:
        ap.error("--image is required when rendering")
    render(cfg, a)


if __name__ == "__main__":
    main()
