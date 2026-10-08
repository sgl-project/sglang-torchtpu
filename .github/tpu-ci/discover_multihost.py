#!/usr/bin/env python3
"""Discover multi-host test files as one matrix row per test file.

A multi-host test file declares a module-level MULTIHOST_LAUNCH dict:
    MULTIHOST_LAUNCH = {"hosts": 2, "launch_script": "recipe/x.sh", "weights_pvc": "...",
                        "schedule": "nightly"}
and registers to a suite via register_tpu_ci(suite=..., nightly=...). This scans
test/registered/**.py (AST only -- no imports/TPU), keeps files registered to --suite
(and nightly when --nightly), then emits each file as its own matrix row so GitHub reports its status directly.

`schedule` is "nightly" for every nightly file; accuracy-test-tpu.yml decides which of its
daily slots runs each accuracy file.

Prints a JSON array for the CI matrix:
    [{"name","launch_script","weights_pvc","hosts","files","schedule"}]  (files = space-separated)
"""

import argparse
import ast
import glob
import json
import os
import sys

from test_names import names_from_path


def _literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _scan(root, suite, nightly_only, launch_var="MULTIHOST_LAUNCH"):
    """{path: launch dict} for files registered to `suite` that declare `launch_var`.

    launch_var=None keeps every registered file, with an empty dict as its spec.
    """
    found = {}
    for path in glob.glob(f"{root}/**/*.py", recursive=True):
        try:
            tree = ast.parse(open(path).read())
        except SyntaxError:
            continue
        launch, registered = None, False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "register_tpu_ci"
            ):
                kw = {k.arg: k.value for k in node.keywords}
                s = _literal(kw["suite"]) if "suite" in kw else None
                n = _literal(kw["nightly"]) if "nightly" in kw else False
                # `disabled` is a skip reason, same as run_suite.py honors for
                # single-host. Without this a disabled multihost test still got
                # a JobSet and a slice every night.
                d = _literal(kw["disabled"]) if "disabled" in kw else None
                if s == suite and (n or not nightly_only):
                    if d is not None:
                        print(f"skip {path}: disabled ({d})", file=sys.stderr)
                        continue
                    registered = True
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == launch_var:
                        launch = _literal(node.value)
        if registered and launch_var is None:
            found[path] = {}
        elif registered and isinstance(launch, dict):
            found[path] = launch
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--root", default="test/registered")
    ap.add_argument("--nightly", action="store_true", default=True)
    a = ap.parse_args()

    out = []
    for path, spec in sorted(_scan(a.root, a.suite, a.nightly).items()):
        try:
            launch_script = spec["launch_script"]
            hosts = int(spec.get("hosts", 2))
            timeout_per_file = int(spec.get("timeout_per_file", 3600))
        except (KeyError, TypeError, ValueError):
            print(f"skip {path}: bad MULTIHOST_LAUNCH {spec!r}", file=sys.stderr)
            continue
        out.append(
            {
                **names_from_path(path),
                "launch_script": launch_script,
                "weights_pvc": spec.get("weights_pvc", "") or "",
                "hosts": hosts,
                "files": path,
                "timeout_per_file": timeout_per_file,
                "schedule": str(spec.get("schedule", "nightly")),
            }
        )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
