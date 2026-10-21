"""TPU P/D /health_generate window: its default and bounds.

Must not come back: a window at or above the transport timeouts, which abort
stuck requests before the window could fire, so the rule would never report
the stall it exists for.

Usage:
    python3 -m pytest test/registered/unit/plugins/test_tpu_pd_health_window.py -v
"""

import unittest
from contextlib import ExitStack

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import patches
from sglang_tpu.disaggregation import health_generate as hg
from sglang_tpu.envs import envs as tpu_envs

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

WINDOW = tpu_envs.SGLANG_TPU_PD_HEALTH_WINDOW_S


class TestPdHealthWindow(CustomTestCase):
    def setUp(self):
        self._stack = ExitStack()
        self.addCleanup(self._stack.close)
        self._stack.enter_context(
            envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.override(300)
        )
        self._stack.enter_context(
            envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.override(250)
        )
        # Start from "unset" and restore whatever the environment had.
        was_set = WINDOW.is_set()
        prior = WINDOW.get()
        WINDOW.clear()
        self.addCleanup(lambda: WINDOW.set(prior) if was_set else WINDOW.clear())

    def test_non_pd_servers_skip_the_check(self):
        with envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.override(10):
            patches.apply_tpu_http_patches(disaggregation_mode="null")

    def test_default_is_a_plain_number_of_seconds_below_the_default_timeouts(self):
        hg.configure_window()
        self.assertEqual(WINDOW.get(), 200.0)
        self.assertEqual(hg.window_s(), 200.0)
        self.assertLess(200.0, envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.default)
        self.assertLess(200.0, envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.default)

    def test_explicit_window_inside_the_bound_is_kept(self):
        WINDOW.set(100)
        hg.configure_window()
        self.assertEqual(WINDOW.get(), 100.0)

    def test_zero_keeps_only_the_probe(self):
        WINDOW.set(0)
        hg.configure_window()
        self.assertEqual(WINDOW.get(), 0.0)
        self.assertIsNone(hg.window_s())

    def test_window_at_or_above_a_transport_timeout_is_refused(self):
        for value in (250, 300, float("inf"), -1):
            WINDOW.set(value)
            with self.subTest(value=value), self.assertRaises(ValueError) as ctx:
                hg.configure_window()
            self.assertIn(
                "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=250", str(ctx.exception)
            )


if __name__ == "__main__":
    unittest.main()
