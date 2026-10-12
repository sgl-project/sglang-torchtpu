#!/usr/bin/env python3
"""Plan the shards of a single-host TPU suite once, for a GitHub matrix.

The shard count is fixed here and handed to every leg as a job output, so a
re-run of a failed leg reuses it: auto_partition is a pure function of the
commit's registrations and the count, and the ref is pinned, hence a re-run
leg runs exactly the files it ran before. Never derive the count inside a leg.

Stdlib only (see ci_register_loader.py); render_jobset.py imports suite_tests.

Usage: plan_partitions.py --suite <name> [--target-seconds S] [--max-partitions N]
Prints {"include": [{"id", "size", "label"}, ...]}; writes the per-shard file
list to $GITHUB_STEP_SUMMARY when set.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Protocol, Sequence, TypedDict

from ci_register_loader import ci_register as _ci_register


class RegisteredTest(Protocol):
    """Fields read here from a ci_register.CIRegistry (loaded without sglang)."""

    filename: str
    est_time: float


class ShardRow(TypedDict):
    """One GitHub matrix leg.

    Attributes:
      id: 0-based shard index, passed as partition_id.
      size: Total shard count, passed as partition_size.
      label: Human-readable "<id + 1>-of-<size>", used in job and JobSet names.
    """

    id: int
    size: int
    label: str


def suite_tests(
    suite: str,
    tests_dir: str = "test/registered",
    *,
    nightly: bool = False,
    test_file: str = "",
) -> list[RegisteredTest]:
    """Collects the enabled TPU tests of a suite, filtered like run_suite.py.

    Args:
      suite: Registered suite name, e.g. stage-a-test-tpu-unit.
      tests_dir: Root scanned recursively for registered test files.
      nightly: Select the suite's nightly registrations instead of per-commit.
      test_file: If set, keep only this file.

    Returns:
      The matching registrations, in discovery order.
    """
    files = [
        path
        for path in glob.glob(os.path.join(tests_dir, "**", "*.py"), recursive=True)
        if os.path.basename(path) not in ("conftest.py", "__init__.py")
    ]
    tests = [
        test
        for test in _ci_register.collect_tests(files, sanity_check=False)
        if test.backend == _ci_register.HWBackend.TPU
        and test.effective_suite == suite
        and test.nightly == nightly
        and test.disabled is None
    ]
    if test_file:
        requested = os.path.normpath(os.path.abspath(test_file))
        tests = [
            test
            for test in tests
            if os.path.normpath(os.path.abspath(test.filename)) == requested
        ]
    return tests


def partition_count(
    tests: Sequence[RegisteredTest], *, target_seconds: float, max_partitions: int
) -> int:
    """Returns the shard count: total est_time / target_seconds, in [1, max]."""
    total = sum(test.est_time for test in tests)
    return max(1, min(max_partitions, math.ceil(total / target_seconds)))


def matrix_rows(size: int) -> list[ShardRow]:
    """Builds the GitHub matrix include list for size shards."""
    return [
        ShardRow(id=rank, size=size, label=f"{rank + 1}-of-{size}")
        for rank in range(size)
    ]


def _write_summary(
    path: str, *, suite: str, shards: Sequence[Sequence[RegisteredTest]]
) -> None:
    """Appends each shard's file list to the step summary as collapsible blocks."""
    size = len(shards)
    with open(path, "a") as f:
        f.write(f"## {suite}: {size} shard(s)\n\n")
        for rank, shard in enumerate(shards):
            est = sum(test.est_time for test in shard)
            f.write(f"<details><summary>{rank + 1}/{size} ")
            f.write(f"({len(shard)} files, est {est:.0f}s)</summary>\n\n")
            for test in sorted(shard, key=lambda t: t.filename):
                f.write(f"- `{test.filename}`\n")
            f.write("\n</details>\n\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--nightly", action="store_true")
    ap.add_argument("--tests-dir", default="test/registered")
    # Registered est_time per shard; the poll and admission budgets scale with it.
    ap.add_argument("--target-seconds", type=float, default=3600)
    # Upper bound on shards; more legs than slices only queue in Kueue.
    ap.add_argument("--max-partitions", type=int, default=4)
    a = ap.parse_args()
    if a.target_seconds <= 0 or a.max_partitions <= 0:
        ap.error("--target-seconds and --max-partitions must be positive")

    tests = suite_tests(a.suite, a.tests_dir, nightly=a.nightly)
    if not tests:
        ap.error(f"suite {a.suite!r} has no enabled TPU tests")
    size = partition_count(
        tests, target_seconds=a.target_seconds, max_partitions=a.max_partitions
    )

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        shards = [
            _ci_register.auto_partition(tests, rank, size) for rank in range(size)
        ]
        _write_summary(summary, suite=a.suite, shards=shards)

    print(json.dumps({"include": matrix_rows(size)}))


if __name__ == "__main__":
    main()
