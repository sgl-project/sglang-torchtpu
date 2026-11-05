"""Tests for scheduler prefill/decode interleaving."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import (
    Scheduler,
    _parse_prefill_decode_interval_adaptive,
)

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_scheduler(
    *,
    interval: int,
    require_mlp_sync: bool,
    skip_idle: bool = False,
    adaptive=None,
) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_decode_interval = interval
    scheduler._prefill_decode_interval_remaining = 0
    scheduler._prefill_decode_interval_elapsed = 0
    scheduler.require_mlp_sync = require_mlp_sync
    scheduler.prefill_decode_interval_skip_idle = skip_idle
    scheduler.prefill_decode_interval_chunked = None
    scheduler.prefill_decode_interval_adaptive = adaptive
    scheduler.chunked_req = None
    scheduler.waiting_queue = []
    return scheduler


def _running(n: int):
    return SimpleNamespace(is_empty=lambda: n == 0, batch_size=lambda: n)


_DECODING = _running(1)
_IDLE = _running(0)


def _make_batch(*, local_extend: bool, global_extend: bool):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: local_extend),
        is_extend_in_batch=global_extend,
    )


class TestPrefillDecodeInterval(unittest.TestCase):
    def test_disabled_interval_does_not_arm(self):
        scheduler = _make_scheduler(interval=0, require_mlp_sync=False)

        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=True, global_extend=False)
        )

        self.assertFalse(scheduler._should_defer_prefill(_DECODING))

    def test_non_dp_interval_uses_local_forward_mode(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=False)

        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=True, global_extend=False)
        )

        self.assertTrue(scheduler._should_defer_prefill(_DECODING))
        self.assertTrue(scheduler._should_defer_prefill(_DECODING))
        self.assertFalse(scheduler._should_defer_prefill(_DECODING))

    def test_dp_interval_uses_globally_synchronized_extend_flag(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=True)

        # This rank is locally decoding, but another DP rank is prefilling.
        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=False, global_extend=True)
        )

        self.assertEqual(scheduler._prefill_decode_interval_remaining, 2)
        self.assertTrue(scheduler._should_defer_prefill(_DECODING))

    def test_skip_idle_does_not_defer_without_a_decode_to_interleave(self):
        # Each deferred loop iteration only idles the device when nothing decodes;
        # the interval resumes counting once requests decode again.
        scheduler = _make_scheduler(interval=3, require_mlp_sync=False, skip_idle=True)
        batch = _make_batch(local_extend=True, global_extend=False)

        scheduler._arm_prefill_decode_interval(batch)
        self.assertFalse(scheduler._should_defer_prefill(_IDLE))
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 0)

        scheduler._arm_prefill_decode_interval(batch)
        self.assertTrue(scheduler._should_defer_prefill(_DECODING))
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 2)

    def test_decode_batch_does_not_rearm_interval(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=True)
        scheduler._prefill_decode_interval_remaining = 1

        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=False, global_extend=False)
        )

        self.assertEqual(scheduler._prefill_decode_interval_remaining, 1)


_EXTEND = _make_batch(local_extend=True, global_extend=False)


class TestAdaptivePrefillDecodeInterval(unittest.TestCase):
    def _armed(self, *, running: int, waiting: int = 1, chunked: bool = False):
        scheduler = _make_scheduler(interval=6, require_mlp_sync=False, adaptive=(3, 2))
        scheduler.waiting_queue = [object()] * waiting
        scheduler.chunked_req = object() if chunked else None
        scheduler._arm_prefill_decode_interval(_EXTEND)
        return scheduler, _running(running)

    def _deferred_steps(self, scheduler, running_batch) -> int:
        steps = 0
        while scheduler._should_defer_prefill(running_batch):
            steps += 1
        return steps

    def test_small_batch_with_waiting_work_ends_after_min_steps(self):
        scheduler, running = self._armed(running=2)
        self.assertEqual(self._deferred_steps(scheduler, running), 2)
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 0)

    def test_batch_at_target_keeps_the_full_interval(self):
        scheduler, running = self._armed(running=3)
        self.assertEqual(self._deferred_steps(scheduler, running), 6)

    def test_nothing_waiting_keeps_the_full_interval(self):
        scheduler, running = self._armed(running=1, waiting=0)
        self.assertEqual(self._deferred_steps(scheduler, running), 6)

    def test_prompt_left_mid_prefill_counts_as_waiting_work(self):
        scheduler, running = self._armed(running=1, waiting=0, chunked=True)
        self.assertEqual(self._deferred_steps(scheduler, running), 2)

    def test_batch_shrinking_below_target_ends_the_interval(self):
        scheduler, _ = self._armed(running=4)
        for _ in range(3):
            self.assertTrue(scheduler._should_defer_prefill(_running(4)))
        self.assertFalse(scheduler._should_defer_prefill(_running(2)))

    def test_rearm_restarts_the_min_steps(self):
        scheduler, running = self._armed(running=2)
        self._deferred_steps(scheduler, running)
        scheduler._arm_prefill_decode_interval(_EXTEND)
        self.assertEqual(scheduler._prefill_decode_interval_remaining, 6)
        self.assertEqual(self._deferred_steps(scheduler, running), 2)

    def test_parse(self):
        self.assertIsNone(_parse_prefill_decode_interval_adaptive(None))
        self.assertIsNone(_parse_prefill_decode_interval_adaptive(""))
        self.assertEqual(_parse_prefill_decode_interval_adaptive("6:8"), (6, 8))
        for bad in ("6", "6:8:1", "a:8", "0:8", "6:-1"):
            with self.assertRaises(ValueError):
                _parse_prefill_decode_interval_adaptive(bad)


if __name__ == "__main__":
    unittest.main()
