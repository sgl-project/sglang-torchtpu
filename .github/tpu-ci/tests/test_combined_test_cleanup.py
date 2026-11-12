"""Check evaluation cleanup without loading models or runtime dependencies."""

import importlib.util
import io
import sys
import unittest
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

ROOT = Path(__file__).resolve().parents[3]


class AccuracyParams:
    name = "gsm8k"
    floor = None


def load_runner():
    accuracy = SimpleNamespace(passed=True, dataset="gsm8k", score=1.0, error=None)
    performance = SimpleNamespace(
        passed=True, output_throughput=None, avg_spec_accept_length=None
    )
    tool_call = SimpleNamespace(passed=True, num_passed=1, num_total=1, failures=[])
    attributes = {
        "sglang": {},
        "sglang.test": {},
        "sglang.test.accuracy_test_runner": {
            "AccuracyTestParams": AccuracyParams,
            "AccuracyTestResult": object,
            "run_accuracy_test": Mock(return_value=accuracy),
            "write_accuracy_github_summary": Mock(),
        },
        "sglang.test.nightly_utils": {"NightlyBenchmarkRunner": Mock()},
        "sglang.test.performance_test_runner": {
            "PerformanceTestParams": object,
            "PerformanceTestResult": object,
            "run_performance_test": Mock(return_value=performance),
        },
        "sglang.test.test_utils": {
            "DEFAULT_URL_FOR_TEST": "http://localhost:30000",
            "ModelLaunchSettings": SimpleNamespace,
            "is_in_ci": lambda: False,
        },
        "sglang.test.tool_call_test_runner": {
            "ToolCallTestParams": object,
            "ToolCallTestResult": object,
            "run_tool_call_test": Mock(return_value=tool_call),
        },
    }
    dependencies = {}
    for name, values in attributes.items():
        dependencies[name] = ModuleType(name)
        vars(dependencies[name]).update(values)
    spec = importlib.util.spec_from_file_location(
        "combined_test_cleanup", ROOT / "python/sglang/test/run_combined_tests.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    module.time = SimpleNamespace(sleep=Mock())
    return module


class TestCombinedTestCleanup(unittest.TestCase):
    def run_tests(self, *launchers, performance=False, accuracy=True, tool_call=False):
        runner = load_runner()
        models = [
            SimpleNamespace(
                model_path=f"model-{index}",
                tp_size=1,
                extra_args=[],
                variant=None,
                launcher=launcher,
            )
            for index, launcher in enumerate(launchers)
        ]
        with redirect_stdout(io.StringIO()):
            result = runner.run_combined_tests(
                models,
                accuracy_params=AccuracyParams() if accuracy else None,
                performance_params=(
                    SimpleNamespace(
                        result_dir=None,
                        batch_sizes=[1],
                        input_lens=[16],
                        output_lens=[8],
                        dataset_name="random",
                        spec_accept_length_threshold=None,
                    )
                    if performance
                    else None
                ),
                tool_call_params=object() if tool_call else None,
            )
        self.assertTrue(result["all_passed"])
        self.assertEqual(
            runner.run_accuracy_test.call_count, len(models) if accuracy else 0
        )
        self.assertEqual(
            runner.run_performance_test.call_count, len(models) if performance else 0
        )
        self.assertEqual(
            runner.run_tool_call_test.call_count, len(models) if tool_call else 0
        )
        return runner.time.sleep

    def test_final_accuracy_phase_does_not_wait(self):
        self.run_tests(None).assert_not_called()

    def test_final_performance_phase_does_not_wait(self):
        self.run_tests(None, performance=True, accuracy=False).assert_not_called()

    def test_final_tool_call_phase_does_not_wait(self):
        self.run_tests(None, accuracy=False, tool_call=True).assert_not_called()

    def test_fixture_owned_server_does_not_wait_before_tool_calls(self):
        self.run_tests(nullcontext, tool_call=True).assert_not_called()

    def test_performance_and_tool_calls_own_servers_with_accuracy_fixtures(self):
        sleep = self.run_tests(
            nullcontext, nullcontext, performance=True, tool_call=True
        )
        self.assertEqual(sleep.call_args_list, [call(20)] * 3)

    def test_performance_waits_before_accuracy_using_a_fixture(self):
        sleep = self.run_tests(nullcontext, performance=True)
        self.assertEqual(sleep.call_args_list, [call(20)])

    def test_tool_calls_wait_between_models_with_accuracy_fixture_launchers(self):
        sleep = self.run_tests(nullcontext, nullcontext, accuracy=False, tool_call=True)
        self.assertEqual(sleep.call_args_list, [call(20)])

    def test_owned_server_waits_before_tool_calls(self):
        sleep = self.run_tests(None, tool_call=True)
        self.assertEqual(sleep.call_args_list, [call(20)])

    def test_owned_server_waits_between_performance_and_tool_calls(self):
        sleep = self.run_tests(None, performance=True, accuracy=False, tool_call=True)
        self.assertEqual(sleep.call_args_list, [call(20)])

    def test_owned_servers_wait_between_models_only(self):
        sleep = self.run_tests(None, None)
        self.assertEqual(sleep.call_args_list, [call(20)])

    def test_owned_servers_wait_between_performance_models_only(self):
        sleep = self.run_tests(None, None, performance=True, accuracy=False)
        self.assertEqual(sleep.call_args_list, [call(20)])

    def test_owned_servers_wait_between_tool_call_models_only(self):
        sleep = self.run_tests(None, None, accuracy=False, tool_call=True)
        self.assertEqual(sleep.call_args_list, [call(20)])

    def test_owned_servers_wait_between_all_phases_and_models(self):
        sleep = self.run_tests(None, None, performance=True, tool_call=True)
        self.assertEqual(sleep.call_args_list, [call(20)] * 5)

    def test_fixture_owned_servers_do_not_wait_between_models(self):
        self.run_tests(nullcontext, nullcontext).assert_not_called()


if __name__ == "__main__":
    unittest.main()
