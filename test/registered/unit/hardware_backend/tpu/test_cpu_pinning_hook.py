"""CPU checks of where the scheduler main-thread pinning hook places threads."""

import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.hooks import cpu_pinning

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MAIN_TID = 100
RANK_CPUS = {14, 15, 16, 126, 127, 128}


class TestCpuPinningHook(CustomTestCase):
    def _run(self, allowed=RANK_CPUS, cpu_count=224, enabled=True):
        placed = {}
        patches = [
            mock.patch.object(
                cpu_pinning.os,
                "sched_getaffinity",
                return_value=set(allowed),
                create=True,
            ),
            mock.patch.object(cpu_pinning.os, "cpu_count", return_value=cpu_count),
            mock.patch.object(
                cpu_pinning.os, "listdir", return_value=["100", "101", "102"]
            ),
            mock.patch.object(
                cpu_pinning.os,
                "sched_setaffinity",
                side_effect=lambda tid, cpus: placed.__setitem__(tid, set(cpus)),
                create=True,
            ),
            mock.patch.object(
                cpu_pinning.threading, "get_native_id", return_value=MAIN_TID
            ),
            mock.patch.object(
                cpu_pinning, "_core_of", side_effect=lambda c: {c, c + 112}
            ),
        ]
        with tpu_envs.SGLANG_TPU_PIN_SCHEDULER_MAIN_THREAD.override(enabled):
            for p in patches:
                p.start()
            try:
                cpu_pinning.pin_scheduler_main_thread(object())
            finally:
                for p in patches:
                    p.stop()
        return placed

    def test_main_thread_gets_a_core_of_its_own(self):
        placed = self._run()
        self.assertEqual(placed[0], {14})
        self.assertEqual(placed[101], {15, 16, 127, 128})
        self.assertEqual(placed[102], {15, 16, 127, 128})
        self.assertNotIn(MAIN_TID, placed)

    def test_off_by_default(self):
        self.assertEqual(self._run(enabled=False), {})

    def test_skipped_without_a_per_rank_cpu_set(self):
        self.assertEqual(self._run(allowed=set(range(224))), {})


if __name__ == "__main__":
    unittest.main()
