"""Every registered test must name a suite its backend declares in run_suite.py.

run_suite.main() validates all registrations before it filters by --hw, so a
single test registered to an undeclared suite makes every backend's run_suite
job fail at startup -- CUDA, AMD, CPU and TPU alike. This turns that into a red
PR on the CPU pr-gate instead.
"""

import importlib.util
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_run_suite():
    path = REPO_ROOT / "test" / "run_suite.py"
    spec = importlib.util.spec_from_file_location("run_suite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRunSuiteRegistrations(CustomTestCase):
    def test_every_registration_names_a_declared_suite(self):
        run_suite = _load_run_suite()
        tests = run_suite.collect_tests(
            run_suite.registered_test_files(), sanity_check=True
        )
        self.assertTrue(tests, "found no registered tests")
        # Raises ValueError naming each file whose suite its backend doesn't declare.
        run_suite.validate_all_suites(tests)


if __name__ == "__main__":
    unittest.main()
