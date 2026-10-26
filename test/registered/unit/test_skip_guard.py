"""Tests for the all-skipped guard that TPU per-commit suites run files through.

Each case writes a tiny test file and runs it for real, through the guard or
through the suite runners, and checks the exit code.
"""

import importlib.util
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci import skip_guard
from sglang.test.ci.chip_parallel import run_files_chip_parallel
from sglang.test.ci.ci_register import HWBackend, register_cpu_ci
from sglang.test.ci.ci_utils import TestFile, run_unittest_files
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_UNITTEST_ALL_SKIPPED = """
    import unittest

    @unittest.skipUnless(False, "needs hardware this host does not have")
    class T(unittest.TestCase):
        def test_a(self):
            pass

        def test_b(self):
            pass

    if __name__ == "__main__":
        unittest.main()
"""

_UNITTEST_SOME_SKIPPED = """
    import unittest

    class T(unittest.TestCase):
        def test_runs(self):
            pass

        @unittest.skip("not here")
        def test_skipped(self):
            pass

    if __name__ == "__main__":
        unittest.main()
"""

_UNITTEST_NONE_SKIPPED = """
    import unittest

    class T(unittest.TestCase):
        def test_runs(self):
            pass

    if __name__ == "__main__":
        unittest.main()
"""

_UNITTEST_FAILING = """
    import unittest

    class T(unittest.TestCase):
        def test_fails(self):
            self.fail("boom")

    if __name__ == "__main__":
        unittest.main()
"""

# SkipTest from setUpClass never reaches testsRun; the file still runs nothing.
_UNITTEST_SETUPCLASS_SKIP = """
    import unittest

    class T(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            raise unittest.SkipTest("no device")

        def test_a(self):
            pass

    if __name__ == "__main__":
        unittest.main()
"""

# One test runs; another skips three subtests. More skip entries than tests
# must not read as "nothing executed".
_UNITTEST_SKIPPED_SUBTESTS = """
    import unittest

    class T(unittest.TestCase):
        def test_runs(self):
            pass

        def test_subtests(self):
            for i in range(3):
                with self.subTest(i=i):
                    self.skipTest("variant unsupported")

    if __name__ == "__main__":
        unittest.main()
"""

_PYTEST_ALL_SKIPPED = """
    import sys

    import pytest

    pytestmark = pytest.mark.skipif(True, reason="needs a TPU")

    def test_a():
        pass

    if __name__ == "__main__":
        sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
"""

_PYTEST_SOME_SKIPPED = """
    import sys

    import pytest

    def test_runs():
        pass

    @pytest.mark.skip(reason="not here")
    def test_skipped():
        pass

    @pytest.mark.xfail(reason="known bug", strict=True)
    def test_xfail():
        assert False

    if __name__ == "__main__":
        sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
"""

# No unittest or pytest run in this process: the guard has no evidence and
# leaves the exit code alone.
_PLAIN_SCRIPT = """
    print("no test framework here")
"""

_SIBLING_IMPORT = """
    import unittest

    from sibling_helper import VALUE

    class T(unittest.TestCase):
        def test_value(self):
            self.assertEqual(VALUE, 42)

    if __name__ == "__main__":
        unittest.main()
"""


def _load_run_suite():
    path = Path(__file__).resolve().parents[2] / "run_suite.py"
    spec = importlib.util.spec_from_file_location("_run_suite_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSkipGuard(CustomTestCase):
    def setUp(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self._dir = Path(tmpdir.name)

    def _write(self, name: str, source: str) -> str:
        path = self._dir / name
        path.write_text(textwrap.dedent(source))
        return str(path)

    def _run(self, source: str, guard: bool) -> subprocess.CompletedProcess:
        path = self._write("test_case_file.py", source)
        command = skip_guard.command_for_test_file(path, fail_on_all_skipped=guard)
        command[0] = sys.executable
        return subprocess.run(command, capture_output=True, text=True, timeout=120)

    def test_exit_codes_through_guard(self):
        no_tests = skip_guard.NO_TESTS_EXECUTED_EXIT_CODE
        cases = {
            "unittest_all_skipped": (_UNITTEST_ALL_SKIPPED, no_tests),
            "unittest_some_skipped": (_UNITTEST_SOME_SKIPPED, 0),
            "unittest_none_skipped": (_UNITTEST_NONE_SKIPPED, 0),
            "unittest_failing": (_UNITTEST_FAILING, 1),
            "unittest_setupclass_skip": (_UNITTEST_SETUPCLASS_SKIP, no_tests),
            "unittest_skipped_subtests": (_UNITTEST_SKIPPED_SUBTESTS, 0),
            "pytest_all_skipped": (_PYTEST_ALL_SKIPPED, no_tests),
            "pytest_some_skipped": (_PYTEST_SOME_SKIPPED, 0),
            "plain_script": (_PLAIN_SCRIPT, 0),
        }
        for name, (source, expected) in cases.items():
            with self.subTest(name):
                completed = self._run(source, guard=True)
                self.assertEqual(
                    completed.returncode, expected, completed.stdout + completed.stderr
                )

    def test_skipped_subtests_without_failfast_are_not_whole_test_skips(self):
        # Without -f every skipped subtest lands in result.skipped (three here),
        # which must not cancel the test that ran.
        path = self._write("test_case_file.py", _UNITTEST_SKIPPED_SUBTESTS)
        command = skip_guard.command_for_test_file(path, fail_on_all_skipped=True)
        command[0] = sys.executable
        command.remove("-f")
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("skipped=3", completed.stderr)

    def test_relative_path_is_resolved(self):
        self._write("test_case_file.py", _UNITTEST_ALL_SKIPPED)
        command = skip_guard.command_for_test_file(
            "test_case_file.py", fail_on_all_skipped=True
        )
        command[0] = sys.executable
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=120, cwd=self._dir
        )
        self.assertEqual(completed.returncode, skip_guard.NO_TESTS_EXECUTED_EXIT_CODE)
        self.assertIn(
            f"[skip_guard] {self._dir.resolve() / 'test_case_file.py'}:",
            completed.stderr,
        )

    def test_message_names_the_file_and_the_ways_out(self):
        completed = self._run(_UNITTEST_ALL_SKIPPED, guard=True)
        self.assertIn("[skip_guard]", completed.stderr)
        self.assertIn("test_case_file.py", completed.stderr)
        self.assertIn("disabled=", completed.stderr)
        self.assertIn("test/manual/", completed.stderr)

    def test_all_skipped_file_passes_without_guard(self):
        completed = self._run(_UNITTEST_ALL_SKIPPED, guard=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("skipped=2", completed.stderr)

    def test_guard_keeps_sibling_imports_working(self):
        self._write("sibling_helper.py", "VALUE = 42\n")
        completed = self._run(_SIBLING_IMPORT, guard=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_run_unittest_files_fails_only_all_skipped_files_when_enabled(self):
        all_skipped = self._write("test_all_skipped.py", _UNITTEST_ALL_SKIPPED)
        some_skipped = self._write("test_some_skipped.py", _UNITTEST_SOME_SKIPPED)
        cases = {
            "guard_on_all_skipped": ([all_skipped], True, -1),
            "guard_on_some_skipped": ([some_skipped], True, 0),
            "guard_off_all_skipped": ([all_skipped], False, 0),
        }
        for name, (files, guard, expected) in cases.items():
            with self.subTest(name):
                result = run_unittest_files(
                    [TestFile(f, estimated_time=5) for f in files],
                    timeout_per_file=120,
                    continue_on_error=True,
                    fail_on_all_skipped=guard,
                )
                self.assertEqual(result, expected)

    def test_chip_parallel_fails_all_skipped_file_when_enabled(self):
        all_skipped = self._write("test_all_skipped.py", _UNITTEST_ALL_SKIPPED)
        some_skipped = self._write("test_some_skipped.py", _UNITTEST_SOME_SKIPPED)
        files = [
            SimpleNamespace(filename=all_skipped, est_time=5, chips=1),
            SimpleNamespace(filename=some_skipped, est_time=5, chips=1),
        ]
        for guard, expected_failed in ((True, [all_skipped]), (False, [])):
            with self.subTest(guard=guard):
                ok, passed, failed, _ = run_files_chip_parallel(
                    files,
                    workers=2,
                    root_dir=str(self._dir / f"chips-{guard}"),
                    timeout_per_file=120,
                    derive_timeout=lambda est: est,
                    continue_on_error=True,
                    fail_on_all_skipped=guard,
                )
                self.assertEqual([name for name, _ in failed], expected_failed)
                self.assertEqual(ok, not expected_failed)

    def test_run_suite_enables_guard_for_tpu_per_commit_suites_only(self):
        run_suite = _load_run_suite()
        cases = {
            "tpu_stage_a": (HWBackend.TPU, ["stage-a-test-tpu-unit"], True),
            "tpu_stage_b": (HWBackend.TPU, ["stage-b-test-tpu-4-v7"], True),
            "tpu_nightly": (HWBackend.TPU, ["nightly-tpu-4-v7"], False),
            "cuda": (HWBackend.CUDA, ["base-b-test-1-gpu-small"], False),
            "cpu": (HWBackend.CPU, ["base-a-test-cpu"], False),
        }
        for name, (hw, suites, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(run_suite.fails_on_all_skipped(hw, suites), expected)


if __name__ == "__main__":
    unittest.main()
