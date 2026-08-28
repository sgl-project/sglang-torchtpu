#!/usr/bin/env python3
"""Emit one GitHub Actions matrix row per enabled TPU test in a suite."""

import argparse
import glob
import importlib.util
import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_CI_REGISTER_PATH = os.path.join(
    REPO_ROOT, "python", "sglang", "test", "ci", "ci_register.py"
)
_spec = importlib.util.spec_from_file_location("tpu_ci_register", _CI_REGISTER_PATH)
_ci_register = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ci_register)


def discover(suite: str, nightly: bool, root: str) -> list[dict]:
    files = [
        path
        for path in glob.glob(os.path.join(root, "**", "*.py"), recursive=True)
        if os.path.basename(path) not in ("conftest.py", "__init__.py")
    ]
    rows = []
    for test in _ci_register.collect_tests(files, sanity_check=False):
        if (
            test.backend != _ci_register.HWBackend.TPU
            or test.effective_suite != suite
            or test.nightly != nightly
            or test.disabled is not None
        ):
            continue
        path = os.path.relpath(test.filename, REPO_ROOT)
        rows.append(
            {
                "name": os.path.splitext(os.path.basename(path))[0],
                "suite": suite,
                "file": path,
            }
        )
    return sorted(rows, key=lambda row: row["name"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--root", default=os.path.join(REPO_ROOT, "test/registered"))
    parser.add_argument("--nightly", action="store_true")
    args = parser.parse_args()
    print(json.dumps(discover(args.suite, args.nightly, args.root)))


if __name__ == "__main__":
    main()
