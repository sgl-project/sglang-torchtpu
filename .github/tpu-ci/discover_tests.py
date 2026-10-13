#!/usr/bin/env python3
"""Emit one GitHub Actions matrix row per enabled TPU test in a suite."""

import argparse
import functools
import glob
import json
import os

from ci_register_loader import ci_register as _ci_register
from test_names import names_from_path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


@functools.lru_cache(maxsize=None)
def _collect(root: str) -> tuple:
    """Every registered test under ``root``, walked and collected once.

    Collecting imports each test file, so it costs seconds; a caller that asks
    about several suites (the dashboard contract lint asks about all of them)
    would otherwise pay that walk per suite.
    """
    files = [
        path
        for path in glob.glob(os.path.join(root, "**", "*.py"), recursive=True)
        if os.path.basename(path) not in ("conftest.py", "__init__.py")
    ]
    return tuple(_ci_register.collect_tests(files, sanity_check=False))


def discover(suite: str, nightly: bool, root: str) -> list[dict]:
    rows = []
    for test in _collect(root):
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
                "suite": suite,
                "file": path,
                "weights_pvc": test.weights_pvc or "",
                **names_from_path(path),
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
