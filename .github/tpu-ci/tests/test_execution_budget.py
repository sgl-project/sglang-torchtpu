"""The whole suite must finish within its configured execution budget."""

import argparse
import ast
import importlib.util
import io
import shlex
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "python/sglang/test/ci/execution_budget.py"


class TestExecutionBudget(unittest.TestCase):
    def setUp(self):
        if not MODULE.exists():
            self.fail("The suite execution-budget checker is not implemented")
        spec = importlib.util.spec_from_file_location("execution_budget", MODULE)
        self.budget = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.budget)

    def check(self, elapsed, minutes=40, code=0):
        output = io.StringIO()
        with (
            patch.object(
                self.budget.time, "monotonic", side_effect=[10000, 10000 + elapsed]
            ),
            redirect_stdout(output),
        ):
            result = self.budget.run_with_execution_budget(lambda: code, minutes)
        return result, output.getvalue()

    def test_exact_limit_passes_and_over_limit_fails(self):
        self.assertEqual(self.check(2400)[0], 0)
        result, output = self.check(2400.1)
        self.assertEqual(result, 1)
        self.assertIn("execution budget exceeded", output)
        self.assertIn("2400.1", output)

    def test_configurable_override(self):
        self.assertEqual(self.check(2600, minutes=45)[0], 0)
        self.assertEqual(self.check(2700.1, minutes=45)[0], 1)

    def test_preserves_failed_suite_exit_code(self):
        for elapsed in (10, 2500):
            self.assertEqual(self.check(elapsed, code=7)[0], 7)

    def test_disabled_budget_preserves_existing_behavior(self):
        with patch.object(
            self.budget.time,
            "monotonic",
            side_effect=AssertionError("timer must stay disabled"),
        ):
            self.assertEqual(self.budget.run_with_execution_budget(lambda: 7, 0), 7)

    def test_invalid_budget_does_not_start_tests(self):
        def must_not_run():
            self.fail("invalid configuration started the suite")

        for minutes in (-1, float("nan"), float("inf")):
            with self.subTest(minutes=minutes), self.assertRaises(ValueError):
                self.budget.run_with_execution_budget(must_not_run, minutes)

    def test_parallel_and_serial_phases_use_one_clock(self):
        clock = [10000.0]

        def suite():
            clock[0] += 1300  # Parallel phase's wall time, not summed worker time.
            clock[0] += 1200  # Serial phase, including loading and precompilation.
            return 0

        with (
            patch.object(self.budget.time, "monotonic", side_effect=lambda: clock[0]),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(self.budget.run_with_execution_budget(suite, 40), 1)

    def test_rendered_command_enforces_budget_through_real_runner_entrypoint(self):
        # Import only main(): the TPU executor needs torch, but its actual CLI
        # and budget wiring can run on CPU with a controlled suite workload.
        tree = ast.parse((ROOT / "test/run_suite.py").read_text())
        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        for minutes, expected in ((40, 1), (45, 0), (0, 0)):
            with self.subTest(minutes=minutes):
                result = subprocess.run(
                    [
                        sys.executable,
                        ".github/tpu-ci/render_jobset.py",
                        "--image",
                        "test",
                        "--suite",
                        "stage-a-test-tpu-unit",
                        "--tests-dir",
                        ".github/tpu-ci/tests",
                        "--tpu-chip-workers",
                        "4",
                        "--execution-budget-minutes",
                        str(minutes),
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                job = yaml.safe_load(result.stdout)
                script = job["spec"]["replicatedJobs"][0]["template"]["spec"][
                    "template"
                ]["spec"]["containers"][0]["args"][0]
                command = shlex.split(
                    next(
                        line.strip()
                        for line in script.splitlines()
                        if "exec python3 test/run_suite.py" in line
                    )
                )
                clock = [10000.0]  # Queue time happened before the runner started.

                def suite(args):
                    self.assertEqual(args.tpu_chip_workers, 4)
                    clock[0] += 1300
                    clock[0] += 1200
                    return 0

                namespace = {
                    "argparse": argparse,
                    "sys": sys,
                    "HW_MAPPING": {"tpu": object()},
                    "run_a_suite": suite,
                    "run_with_execution_budget": self.budget.run_with_execution_budget,
                }
                module = ast.Module(body=[main], type_ignores=[])
                exec(compile(module, "test/run_suite.py", "exec"), namespace)
                with (
                    patch.object(sys, "argv", command[2:]),
                    patch.object(
                        self.budget.time, "monotonic", side_effect=lambda: clock[0]
                    ),
                    redirect_stdout(io.StringIO()),
                    self.assertRaises(SystemExit) as exited,
                ):
                    namespace["main"]()
                self.assertEqual(exited.exception.code, expected)


if __name__ == "__main__":
    unittest.main()
