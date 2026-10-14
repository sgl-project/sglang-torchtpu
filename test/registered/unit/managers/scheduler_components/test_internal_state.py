import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.scheduler_components import internal_state
from sglang.srt.managers.scheduler_components.internal_state import (
    SchedulerInternalStateController,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCompilationCacheMisses(CustomTestCase):
    def setUp(self):
        super().setUp()
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(
                unsafe_save_torch_compile_bundle=None,
                unsafe_load_torch_compile_bundle=None,
            )
        )
        self.controller = SchedulerInternalStateController(
            scheduler, record_step_time=False
        )
        execution = SimpleNamespace(graph=SimpleNamespace(enable_torch_compile=False))
        patcher = patch.object(internal_state, "get_exec", return_value=execution)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_reports_platform_compilation_cache_misses(self):
        state = {}
        with patch.object(
            internal_state,
            "current_platform",
            SimpleNamespace(get_compilation_cache_misses=lambda: 37),
        ):
            self.controller._add_torch_compile(state)

        self.assertEqual(state["compilation_cache_misses"], 37)

    def test_omits_counter_when_platform_does_not_expose_it(self):
        state = {}
        with patch.object(
            internal_state,
            "current_platform",
            SimpleNamespace(get_compilation_cache_misses=lambda: None),
        ):
            self.controller._add_torch_compile(state)

        self.assertNotIn("compilation_cache_misses", state)


if __name__ == "__main__":
    unittest.main()
