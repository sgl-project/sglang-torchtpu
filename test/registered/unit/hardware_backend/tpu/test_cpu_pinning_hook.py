"""CPU checks of where the scheduler main-thread pinning hook places threads."""

import os
import tempfile
import types
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
    def _run(
        self,
        allowed=RANK_CPUS,
        cpu_count=224,
        enabled=True,
        busy_map=None,
        core_of_fn=None,
    ):
        placed = {}
        if core_of_fn is None:
            core_of_fn = lambda c: {c, c + 112}
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
            mock.patch.object(cpu_pinning, "_core_of", side_effect=core_of_fn),
            mock.patch.object(
                cpu_pinning,
                "_measure_cpu_busy_fraction",
                return_value=dict(busy_map or {}),
            ),
        ]
        with tpu_envs.SGLANG_TPU_ENABLE_SCHEDULER_THREAD_PINNING.override(enabled):
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

    def test_pin_skips_saturated_lowest_numbered_core(self):
        placed = self._run(
            allowed={60, 61, 62, 120, 121, 122},
            busy_map={60: 1.0, 120: 0.0},
            core_of_fn=lambda c: {c, c + 60} if c < 120 else {c - 60, c},
        )
        self.assertEqual(placed[0], {61})
        self.assertEqual(placed[101], {62, 122})
        self.assertEqual(placed[102], {62, 122})
        self.assertNotIn(60, placed[101])
        self.assertNotIn(120, placed[101])

    def test_pin_falls_back_to_clean_pool_when_all_cores_busy(self):
        busy_map = {c: 0.40 for c in RANK_CPUS}
        placed = self._run(busy_map=busy_map)
        self.assertNotIn(0, placed)
        self.assertEqual(placed[MAIN_TID], RANK_CPUS)
        self.assertEqual(placed[101], RANK_CPUS)
        self.assertEqual(placed[102], RANK_CPUS)

    def test_measure_cpu_busy_fraction_parses_proc_stat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stat_path = os.path.join(tmpdir, "stat")
            with open(stat_path, "w") as f:
                f.write("cpu  100 0 100 200 0 0 0 0 0 0\n")
                f.write("cpu60 100 0 0 100 0 0 0 0 0 0\n")
                f.write("cpu61 50 0 0 50 0 0 0 0 0 0\n")
            first = cpu_pinning._read_proc_stat_cpu_times(stat_path)
            self.assertEqual(first[60], (100, 200))
            self.assertEqual(first[61], (50, 100))

            with open(stat_path, "w") as f:
                f.write("cpu60 200 0 0 100 0 0 0 0 0 0\n")
                f.write("cpu61 75 0 0 125 0 0 0 0 0 0\n")
            second = cpu_pinning._read_proc_stat_cpu_times(stat_path)

        with mock.patch.object(
            cpu_pinning, "_read_proc_stat_cpu_times", side_effect=[first, second]
        ):
            busy = cpu_pinning._measure_cpu_busy_fraction(
                {60, 61}, sample_seconds=0.0, stat_path="/unused"
            )
        self.assertAlmostEqual(busy[60], 1.0)
        self.assertAlmostEqual(busy[61], 0.25)

    def test_bind_rank_cpu_set_cgroup_fallback_uses_psutil(self):
        import psutil

        from sglang.srt import runtime_context

        placed = {}
        parallel = types.SimpleNamespace(tp_size=8, tp_rank=1, nnodes=1, pp_size=1)
        original_fn = mock.Mock(side_effect=OSError("cgroup restricted"))

        with (
            mock.patch.object(runtime_context, "get_parallel", return_value=parallel),
            mock.patch.object(
                cpu_pinning.os, "sched_getaffinity", return_value={1, 2, 9}, create=True
            ),
            mock.patch.object(
                cpu_pinning.os,
                "sched_setaffinity",
                side_effect=lambda tid, cpus: placed.__setitem__(tid, set(cpus)),
                create=True,
            ),
            mock.patch.object(
                psutil,
                "cpu_count",
                side_effect=lambda logical=True: 16 if logical else 8,
            ),
        ):
            cpu_pinning.bind_rank_cpu_set(original_fn, gpu_id=0)

        original_fn.assert_called_once_with(1)
        self.assertEqual(placed[0], {1, 9})


if __name__ == "__main__":
    unittest.main()
