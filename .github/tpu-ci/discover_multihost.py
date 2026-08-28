#!/usr/bin/env python3
"""Discover multi-host test files and group them by launch spec (for the nightly matrix).

A multi-host test file declares a module-level MULTIHOST_LAUNCH dict:
    MULTIHOST_LAUNCH = {"hosts": 2, "launch_script": "recipe/x.sh", "weights_pvc": "..."}
and registers to a suite via register_tpu_ci(suite=..., nightly=...). This scans
test/registered/**.py (AST only -- no imports/TPU), keeps files registered to --suite
(and nightly when --nightly), then GROUPS files whose (launch_script, weights_pvc, hosts)
are identical into ONE launch group -- so same-setup files share one launched server.

Prints a JSON array for the CI matrix:
    [{"name","launch_script","weights_pvc","hosts","files"}]   (files = space-separated)
"""

import argparse
import ast
import glob
import json
import os
import re
import sys


def _literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _scan(root, suite, nightly_only):
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
                    if isinstance(t, ast.Name) and t.id == "MULTIHOST_LAUNCH":
                        launch = _literal(node.value)
        if registered and isinstance(launch, dict):
            found[path] = launch
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--root", default="test/registered")
    ap.add_argument("--nightly", action="store_true", default=True)
    a = ap.parse_args()

    groups = {}
    for path, spec in sorted(_scan(a.root, a.suite, a.nightly).items()):
        try:
            ls, hosts = spec["launch_script"], int(spec.get("hosts", 2))
        except (KeyError, TypeError, ValueError):
            print(f"skip {path}: bad MULTIHOST_LAUNCH {spec!r}", file=sys.stderr)
            continue
        wp = spec.get("weights_pvc", "") or ""
        key = (ls, wp, hosts)
        entry = groups.setdefault(key, {"files": [], "timeout_per_file": 3600})
        entry["files"].append(path)
        # A group runs one `timeout` per file (jobset-tpu.yaml); the slowest
        # file's declared budget bounds them all.
        entry["timeout_per_file"] = max(
            entry["timeout_per_file"], int(spec.get("timeout_per_file", 3600))
        )

    out = []
    for (ls, wp, hosts), entry in groups.items():
        files = entry["files"]
        # Recipes are named *_mh.sh; drop that redundant suffix so the job reads
        # mh-<model> rather than mh-<model>-mh.
        base = os.path.basename(ls).lower().removesuffix(".sh").removesuffix("_mh")
        slug = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
        out.append(
            {
                "name": f"mh-{slug}",
                "launch_script": ls,
                "weights_pvc": wp,
                "hosts": hosts,
                "files": " ".join(sorted(files)),
                "timeout_per_file": entry["timeout_per_file"],
            }
        )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
