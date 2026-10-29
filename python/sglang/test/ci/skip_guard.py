"""Fail a registered test file when none of its tests execute.

The suite runner launches each file as ``python3 <file> -f`` and trusts the
exit code. unittest and pytest both exit 0 when every case skips, so a file
whose gate never opens on the CI host (for example ``jax.devices()`` showing
no TPU) reports green without testing anything. See
sgl-project/sglang-torchtpu#413.

``python3 skip_guard.py <file> [args...]`` runs the file as ``__main__`` the way
the interpreter would, records every unittest run and pytest session in that
process, and exits with ``NO_TESTS_EXECUTED_EXIT_CODE`` if the file exited 0
after at least one run in which no test executed. A file with no unittest or
pytest run in its own process (it shells out, or uses a custom runner that
does not derive from ``unittest.TextTestRunner``) keeps its own exit code.

Standard library only: the runner launches this module by path, so it must not
depend on the package that contains it being importable.
"""

import contextlib
import os
import runpy
import sys
import unittest
from typing import Iterator, List, Optional, Sequence

# Same value as unittest's "no tests ran" (Python 3.12+) and pytest's
# ExitCode.NO_TESTS_COLLECTED.
NO_TESTS_EXECUTED_EXIT_CODE = 5

_GUARD_PATH = os.path.abspath(__file__)


def command_for_test_file(path: str, fail_on_all_skipped: bool) -> List[str]:
    """The command the suite runners use to execute one test file."""
    if fail_on_all_skipped:
        return ["python3", _GUARD_PATH, path, "-f"]
    return ["python3", path, "-f"]


class _Tally:
    """Test runs seen in this process and the tests that actually executed."""

    def __init__(self) -> None:
        self.runs = 0
        self.executed = 0


def _unittest_executed(result: unittest.TestResult) -> int:
    # result.skipped also holds skipped subtests (one test can have many) and
    # _ErrorHolder entries for a SkipTest raised in setUpClass/setUpModule
    # (never counted in testsRun). Only a whole test method skipped by its
    # decorator or a skipTest() call cancels one unit of testsRun. Skipped
    # subtests arrive as unittest's private _SubTest, a TestCase subclass:
    # without excluding them, one passing test plus a test that skips three
    # subtests reads as -1 executed. (With -f, unittest stops a subtest loop at
    # its first skip, so only one entry appears.)
    whole_test_skips = sum(
        1
        for test, _ in result.skipped
        if isinstance(test, unittest.TestCase)
        and not isinstance(test, unittest.case._SubTest)
    )
    return result.testsRun - whole_test_skips


class _PytestTally:
    """pytest plugin feeding a _Tally."""

    def __init__(self, tally: _Tally) -> None:
        self._tally = tally

    def pytest_sessionstart(self, session) -> None:
        self._tally.runs += 1

    def pytest_runtest_logreport(self, report) -> None:
        # pytest reports an xfail as skipped, but its body did run.
        if report.when == "call" and (
            not report.skipped or hasattr(report, "wasxfail")
        ):
            self._tally.executed += 1


@contextlib.contextmanager
def _recording(tally: _Tally) -> Iterator[None]:
    original_run = unittest.TextTestRunner.run

    def run(self, test):
        result = original_run(self, test)
        tally.runs += 1
        tally.executed += _unittest_executed(result)
        return result

    unittest.TextTestRunner.run = run
    try:
        import pytest
    except ImportError:
        pytest = None
    if pytest is not None:
        original_main = pytest.main

        def main(args=None, plugins=None):
            return original_main(args, [*(plugins or ()), _PytestTally(tally)])

        pytest.main = main
    try:
        yield
    finally:
        unittest.TextTestRunner.run = original_run
        if pytest is not None:
            pytest.main = original_main


def _exit_code(code) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr, flush=True)
    return 1


def run_test_file(path: str, args: Sequence[str]) -> int:
    """Run ``path`` as ``__main__`` and return its exit code, failing it when
    a test run inside it executed no test."""
    # Suite runners may pass a relative path; resolve it once so the message
    # and sys.path do not depend on the working directory.
    path = os.path.abspath(path)
    sys.argv = [path, *args]
    # Match ``python3 <path>``: tests import helper modules next to them.
    sys.path[0] = os.path.dirname(path)
    tally = _Tally()
    with _recording(tally):
        try:
            runpy.run_path(path, run_name="__main__")
            code = 0
        except SystemExit as exc:
            code = _exit_code(exc.code)
    if code == 0 and tally.runs and tally.executed <= 0:
        print(
            f"\n[skip_guard] {path}: no test executed in {tally.runs} test "
            "run(s); every case skipped. Make its gate pass on this CI host, "
            'or register it with disabled="<reason>", or move it to '
            "test/manual/.",
            file=sys.stderr,
            flush=True,
        )
        return NO_TESTS_EXECUTED_EXIT_CODE
    return code


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(f"usage: {sys.argv[0]} <test_file> [args...]", file=sys.stderr)
        return 2
    return run_test_file(argv[0], argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
