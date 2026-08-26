#!/usr/bin/env python3
"""Render the Meta (Project B) multi-host JobSet (jobset-meta-tpu.yaml), and discover the
`meta-nightly-*` launch groups for the nightly matrix. Single source of truth for rendering
onto Meta's external TPU cluster -- used by _meta-tpu-test.yml and runnable locally for
dry-runs. See meta-topology.yaml + README.md.

A Meta multi-host test declares a module-level MULTIHOST_LAUNCH dict and registers to a
meta suite:

    MULTIHOST_LAUNCH = {
        "launch_script": "recipe/qwen3_coder_30b_a3b_fp8.sh",  # relative to benchmark/tpu/
        "weights": "Qwen3-Coder-30B-A3B-Instruct-FP8",         # subdir under gs://<bucket>/weights/
        # "extra_args": "...",   # optional; overrides weights.default_extra_args
    }
    register_tpu_ci(est_time=1200, suite="meta-nightly-2x2x2", nightly=True)

The SUITE fixes the slice (hosts/topology/nodepool live in meta-topology.yaml), so a test
picks its slice purely by suite name. tp = hosts x devices_per_host.

Modes:
  --list-groups [--nightly]
      Scan test/registered/**.py (AST only -- no imports/TPU), keep files registered to any
      meta suite in meta-topology.yaml, GROUP files with identical (suite, launch_script,
      weights) into one launch group, and print a JSON matrix for the CI:
        [{"name","suite","hosts","topology","nodepool_var","launch_script","weights","files"}]
      `nodepool_var` is the NAME of the repo secret holding that slice's nodepool; the
      workflow resolves it so the value never appears in source or the matrix.
  render (default)
      --image <ref> --suite <meta-suite> --launch-script <rel> --files "<f>..." \
          --weights <model-subdir> --nodepool <value> [--name <run-name>] [--extra-args "<flags>"]

Usage: render_meta_jobset.py --list-groups
       render_meta_jobset.py --image <ref> --suite meta-nightly-2x2x2 --launch-script recipe/x.sh \
           --files "a.py b.py" --weights <model-subdir> --nodepool <value> > job.yaml
"""

import argparse
import ast
import glob
import json
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

# ~30m bring-up (weight load over gcsfuse + compile warmup) + rank-0's test files + margin.
_ACTIVE_DEADLINE = 9000


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def dns_safe(name):
    s = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    return (s[:40].strip("-")) or "run"


def _literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _scan(root, suites, nightly_only):
    """Return {path: (suite, MULTIHOST_LAUNCH)} for files registered to any meta suite."""
    found = {}
    for path in glob.glob(f"{root}/**/*.py", recursive=True):
        try:
            tree = ast.parse(open(path).read())
        except SyntaxError:
            continue
        launch, matched_suite = None, None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "register_tpu_ci"
            ):
                kw = {k.arg: k.value for k in node.keywords}
                s = _literal(kw["suite"]) if "suite" in kw else None
                n = _literal(kw["nightly"]) if "nightly" in kw else False
                if s in suites and (n or not nightly_only):
                    matched_suite = s
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "MULTIHOST_LAUNCH":
                        launch = _literal(node.value)
        if matched_suite and isinstance(launch, dict):
            found[path] = (matched_suite, launch)
    return found


def list_groups(cfg, root, nightly_only):
    suites = cfg["suites"]
    groups = {}
    for path, (suite, spec) in sorted(_scan(root, suites, nightly_only).items()):
        ls = spec.get("launch_script")
        weights = spec.get("weights")
        if not ls or not weights:
            print(
                f"skip {path}: MULTIHOST_LAUNCH needs launch_script + weights, got {spec!r}",
                file=sys.stderr,
            )
            continue
        key = (suite, ls, weights)
        groups.setdefault(key, []).append(path)

    out = []
    for (suite, ls, weights), files in groups.items():
        sl = suites[suite]
        slug = re.sub(
            r"[^a-z0-9]+", "-", os.path.basename(ls).lower().removesuffix(".sh")
        ).strip("-")
        out.append(
            {
                "name": f"meta-{slug}",
                "suite": suite,
                "hosts": sl["hosts"],
                "topology": sl["topology"],
                # The NAME of the repo secret holding this slice's nodepool; the workflow
                # resolves it. The nodepool value never appears in source or the matrix.
                "nodepool_var": sl["nodepool_var"],
                "launch_script": ls,
                "weights": weights,
                "files": " ".join(sorted(files)),
            }
        )
    return out


def render(cfg, a):
    suites = cfg["suites"]
    if a.suite not in suites:
        sys.exit(f"unknown --suite {a.suite!r}; meta-topology.yaml has {sorted(suites)}")
    if not a.launch_script or not a.files or not a.weights:
        sys.exit("render needs --launch-script, --files and --weights")

    if not a.nodepool:
        sys.exit("render needs --nodepool (the resolved slice nodepool, from a repo secret)")

    sl = suites[a.suite]
    cl = cfg["cluster"]
    w = cfg["weights"]
    hosts = int(sl["hosts"])
    tp = hosts * int(cl["devices_per_host"])
    model_path = f"{w['mount_path']}/{w['prefix']}/{a.weights}"
    extra_args = a.extra_args or w.get("default_extra_args", "")

    vals = {
        "IMAGE": a.image,
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
    tpl = open(f"{HERE}/jobset-meta-tpu.yaml").read()
    out = re.sub(r"\$\{(\w+)\}", lambda m: vals.get(m.group(1), m.group(0)), tpl)
    sys.stdout.write(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=f"{HERE}/meta-topology.yaml")
    ap.add_argument("--tests-dir", default="test/registered")
    ap.add_argument(
        "--list-groups",
        action="store_true",
        help="Print the JSON launch-group matrix for the meta suites and exit.",
    )
    ap.add_argument("--nightly", action="store_true", default=True)
    # render mode
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
