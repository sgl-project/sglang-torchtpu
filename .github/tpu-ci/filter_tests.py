#!/usr/bin/env python3
"""Select the tests matching a path substring.

Two modes, both taking the same `--filter` substring:

  (default) Read a JSON row list on stdin, write the kept rows to stdout, so it
            composes as a pipe stage in a discover step. Shape-agnostic: a
            discover_tests.py row names its test in `file`, a
            discover_multihost.py launch group in `files`.
  --verify  Exit non-zero unless the filter matches some test the feature
            validation stage can run. Nothing downstream can catch a typo: an
            empty matrix makes GitHub skip its jobs, and a skipped job reads as
            a pass, so a filter matching nothing would otherwise produce a
            green run that tested nothing.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TPU_CI = REPO / ".github" / "tpu-ci"
FEATURE_VALIDATION = REPO / ".github" / "workflows" / "feature-validation-tpu.yml"
# The static P/D and diffusion jobs hand-wire their one file rather than being
# discovered. Same canonical source test_tpu_dashboard_contract.py reads them
# from -- the matrix jobs pass an expression here, so they do not match.
_STATIC_FILES = re.compile(r"^\s+files:\s+(\S+\.py)\s*$", re.M)


def _row_path(row: dict) -> str:
    """The row's test file: `file` on a test row, `files` on a launch group."""
    return row.get("file") or row.get("files", "")


def _filter_rows(rows: list[dict], needle: str) -> list[dict]:
    return [row for row in rows if needle in _row_path(row)]


def _discover(script: str, *args: str) -> list[dict]:
    completed = subprocess.run(
        [sys.executable, str(TPU_CI / script), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        check=True,
    )
    return json.loads(completed.stdout)


def _external_group_files() -> list[str]:
    """Files of the external-cluster nightly suites, one entry per file."""
    return [
        file
        for row in _discover(
            "external/render_external_jobset.py", "--list-groups", "--nightly"
        )
        for group in row["groups"]
        for file in group["files"].split()
    ]


def _stage_test_files() -> list[str]:
    """Every test file feature validation can run: the in-project single-host
    matrix and multi-host groups, the hand-wired static jobs, and the
    external-cluster suites."""
    rows = _discover("discover_tests.py", "--suite", "nightly-tpu-4-v7", "--nightly")
    rows += _discover("discover_multihost.py", "--suite", "nightly-tpu-multihost")
    return (
        [_row_path(row) for row in rows]
        + _STATIC_FILES.findall(FEATURE_VALIDATION.read_text(encoding="utf-8"))
        + _external_group_files()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", required=True, help="substring of a test file path")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check the filter matches a runnable test instead of filtering stdin",
    )
    args = parser.parse_args()
    if args.verify:
        files = _stage_test_files()
        matched = [path for path in files if args.filter in path]
        if not matched:
            sys.exit(
                f"filter {args.filter!r} matched none of the {len(files)} tests "
                f"feature validation can run"
            )
        print("\n".join([f"filter {args.filter!r} matches:", *matched]))
        return
    json.dump(_filter_rows(json.load(sys.stdin), args.filter), sys.stdout)


if __name__ == "__main__":
    main()
