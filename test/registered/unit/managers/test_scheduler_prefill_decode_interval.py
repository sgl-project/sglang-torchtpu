"""Tests for scheduler prefill/decode interleaving."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_scheduler(
    *, interval: int, require_mlp_sync: bool, skip_idle: bool = False
) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_decode_interval = interval
    scheduler._prefill_decode_interval_remaining = 0
    scheduler.require_mlp_sync = require_mlp_sync
    scheduler.prefill_decode_interval_skip_idle = skip_idle
    return scheduler


_DECODING = SimpleNamespace(is_empty=lambda: False)
_IDLE = SimpleNamespace(is_empty=lambda: True)


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


if __name__ == "__main__":
    unittest.main()
