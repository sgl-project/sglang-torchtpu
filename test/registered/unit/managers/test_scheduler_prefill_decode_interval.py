"""Tests for scheduler prefill/decode interleaving."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import NextBatchPlan
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import reset_context
from sglang_tpu.hooks import HOOKS as TPU_HOOKS
from sglang_tpu.hooks import HookType
from sglang_tpu.hooks.scheduler import configure_tpu_scheduler

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_scheduler(
    *,
    interval: int,
    require_mlp_sync: bool,
    tpu_drain_interval: int = 0,
    waiting_queue: list | None = None,
) -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_decode_interval = interval
    scheduler._prefill_decode_interval_remaining = 0
    scheduler.require_mlp_sync = require_mlp_sync
    scheduler._tpu_prefill_drain_interval = tpu_drain_interval
    scheduler._tpu_drain_step_counter = 0
    scheduler._tpu_drain_prefill = False
    scheduler.chunked_req = None
    scheduler.last_batch = None
    scheduler.waiting_queue = [] if waiting_queue is None else list(waiting_queue)
    return scheduler


def _make_batch(*, local_extend: bool, global_extend: bool, empty: bool = False):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: local_extend),
        is_extend_in_batch=global_extend,
        is_empty=lambda: empty,
        is_prefill_only=False,
        batch_is_full=False,
    )


def _make_full_scheduler(*, tpu_drain_interval: int, running_batch, waiting_queue):
    reset_context()
    scheduler = _make_scheduler(
        interval=0,
        require_mlp_sync=True,
        tpu_drain_interval=tpu_drain_interval,
        waiting_queue=waiting_queue,
    )
    scheduler.scheduler_stage_metrics = None
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.dllm_config = None
    scheduler.dllm_manager = None
    scheduler.enable_hisparse = False
    scheduler.enable_fpm = False
    scheduler.enable_hierarchical_cache = False
    scheduler.enable_hicache_storage = False
    scheduler.enable_unified_cache_external_linker = False
    scheduler.enable_lmcache = False
    scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    scheduler._pending_chunked_abort_req = None
    scheduler._deferred_chunked_sender_aborts = []
    scheduler.process_pending_chunked_abort = lambda: None
    scheduler.running_batch = running_batch
    scheduler.get_new_batch_prefill = MagicMock(
        return_value=NextBatchPlan(batch_to_run=None, running_batch=running_batch)
    )
    scheduler.update_running_batch = MagicMock(side_effect=lambda batch: batch)
    scheduler.dp_attn_adapter = SimpleNamespace(
        maybe_prepare_mlp_sync_batch=lambda batch, **_: batch,
        maybe_convert_decode_to_extend=lambda batch: batch,
    )
    scheduler.ngram_embedding_manager = SimpleNamespace(
        prepare_for_forward=lambda batch, **_: batch
    )
    return scheduler


class TestPrefillDecodeInterval(unittest.TestCase):
    def test_disabled_interval_does_not_arm(self):
        scheduler = _make_scheduler(interval=0, require_mlp_sync=False)

        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=True, global_extend=False)
        )

        self.assertFalse(scheduler._should_defer_prefill())

    def test_non_dp_interval_uses_local_forward_mode(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=False)

        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=True, global_extend=False)
        )

        self.assertTrue(scheduler._should_defer_prefill())
        self.assertTrue(scheduler._should_defer_prefill())
        self.assertFalse(scheduler._should_defer_prefill())

    def test_dp_interval_uses_globally_synchronized_extend_flag(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=True)

        # This rank is locally decoding, but another DP rank is prefilling.
        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=False, global_extend=True)
        )

        self.assertEqual(scheduler._prefill_decode_interval_remaining, 2)
        self.assertTrue(scheduler._should_defer_prefill())

    def test_decode_batch_does_not_rearm_interval(self):
        scheduler = _make_scheduler(interval=2, require_mlp_sync=True)
        scheduler._prefill_decode_interval_remaining = 1

        scheduler._arm_prefill_decode_interval(
            _make_batch(local_extend=False, global_extend=False)
        )

        self.assertEqual(scheduler._prefill_decode_interval_remaining, 1)


class TestTpuPrefillDrainInterval(unittest.TestCase):
    def test_periodic_drain_opens_drains_closes_and_defers_between_boundaries(self):
        scheduler = _make_scheduler(
            interval=0,
            require_mlp_sync=True,
            tpu_drain_interval=4,
            waiting_queue=["req-1"],
        )
        running_batch = _make_batch(
            local_extend=False, global_extend=False, empty=False
        )
        scheduler.last_batch = running_batch

        # Step 0 opens the periodic drain window and keeps admitting while queued.
        self.assertFalse(scheduler._should_defer_prefill(running_batch))
        self.assertTrue(scheduler._tpu_drain_prefill)
        scheduler._tpu_drain_step_counter = 1
        self.assertFalse(scheduler._should_defer_prefill(running_batch))

        # Draining the waiting queue closes the window; mid-window arrivals wait.
        scheduler.waiting_queue.clear()
        self.assertTrue(scheduler._should_defer_prefill(running_batch))
        self.assertFalse(scheduler._tpu_drain_prefill)
        scheduler.waiting_queue.append("req-2")
        scheduler._tpu_drain_step_counter = 2
        self.assertTrue(scheduler._should_defer_prefill(running_batch))

        # Step 4 re-opens the drain window across ranks together.
        scheduler._tpu_drain_step_counter = 4
        self.assertFalse(scheduler._should_defer_prefill(running_batch))
        self.assertTrue(scheduler._tpu_drain_prefill)

    def test_chunked_prefill_continuation_is_never_deferred(self):
        scheduler = _make_scheduler(
            interval=0,
            require_mlp_sync=True,
            tpu_drain_interval=8,
            waiting_queue=[],
        )
        scheduler._tpu_drain_step_counter = 3
        scheduler._tpu_drain_prefill = False
        scheduler.chunked_req = SimpleNamespace(rid="chunked-req")
        running_batch = _make_batch(
            local_extend=False, global_extend=False, empty=False
        )
        scheduler.last_batch = running_batch

        self.assertFalse(scheduler._should_defer_prefill(running_batch))

    def test_empty_rank_joins_wave_only_when_dp_peers_are_idle_or_extending(self):
        empty_running = _make_batch(local_extend=False, global_extend=False, empty=True)

        # Peer DP ranks are decoding (`is_extend_in_batch=False`), so this empty rank
        # stays synchronized and defers until the periodic drain boundary.
        scheduler = _make_scheduler(
            interval=0,
            require_mlp_sync=True,
            tpu_drain_interval=4,
            waiting_queue=["req-1"],
        )
        scheduler._tpu_drain_step_counter = 1
        scheduler.last_batch = _make_batch(local_extend=False, global_extend=True)
        scheduler.last_batch.is_extend_in_batch = False
        self.assertTrue(scheduler._should_defer_prefill(empty_running))
        self.assertFalse(scheduler._tpu_drain_prefill)

        # Peer DP rank ran EXTEND (`is_extend_in_batch=True`): empty rank joins wave.
        scheduler.last_batch.is_extend_in_batch = True
        self.assertFalse(scheduler._should_defer_prefill(empty_running))
        self.assertTrue(scheduler._tpu_drain_prefill)

        # Starting from idle (`last_batch=None`): empty rank opens wave immediately.
        scheduler_idle = _make_scheduler(
            interval=0,
            require_mlp_sync=True,
            tpu_drain_interval=4,
            waiting_queue=["req-1"],
        )
        scheduler_idle._tpu_drain_step_counter = 2
        self.assertFalse(scheduler_idle._should_defer_prefill(empty_running))
        self.assertTrue(scheduler_idle._tpu_drain_prefill)

        # Non-DP (`require_mlp_sync=False`): empty rank admits without peer check.
        scheduler_non_dp = _make_scheduler(
            interval=0,
            require_mlp_sync=False,
            tpu_drain_interval=4,
            waiting_queue=["req-1"],
        )
        scheduler_non_dp._tpu_drain_step_counter = 2
        scheduler_non_dp.last_batch = _make_batch(
            local_extend=False, global_extend=False
        )
        self.assertFalse(scheduler_non_dp._should_defer_prefill(empty_running))
        self.assertTrue(scheduler_non_dp._tpu_drain_prefill)

    def test_get_next_batch_closes_capacity_blocked_drain_and_resets_on_idle(self):
        running_batch = _make_batch(
            local_extend=False, global_extend=False, empty=False
        )
        scheduler = _make_full_scheduler(
            tpu_drain_interval=4,
            running_batch=running_batch,
            waiting_queue=["blocked-req"],
        )
        scheduler.last_batch = running_batch

        with (
            patch(
                "sglang.srt.managers.scheduler.get_memory",
                return_value=SimpleNamespace(enable_flexkv=False),
            ),
            patch("sglang.srt.managers.scheduler.set_schedule_time_batch"),
        ):
            plan = scheduler.get_next_batch_to_run(running_batch, running_batch)

        self.assertIs(plan.batch_to_run, running_batch)
        scheduler.get_new_batch_prefill.assert_called_once()
        # Unable to admit any queued request due to capacity: closes drain before next decode.
        self.assertFalse(scheduler._tpu_drain_prefill)
        self.assertEqual(scheduler._tpu_drain_step_counter, 1)

        # Idle step (no running decode and no admitted prefill) resets step counter.
        empty_batch = _make_batch(local_extend=False, global_extend=False, empty=True)
        idle_scheduler = _make_full_scheduler(
            tpu_drain_interval=4,
            running_batch=empty_batch,
            waiting_queue=[],
        )
        idle_scheduler._tpu_drain_step_counter = 17
        idle_scheduler._tpu_drain_prefill = True

        with (
            patch(
                "sglang.srt.managers.scheduler.get_memory",
                return_value=SimpleNamespace(enable_flexkv=False),
            ),
            patch("sglang.srt.managers.scheduler.set_schedule_time_batch"),
        ):
            idle_plan = idle_scheduler.get_next_batch_to_run(empty_batch, None)

        self.assertIsNone(idle_plan.batch_to_run)
        self.assertEqual(idle_scheduler._tpu_drain_step_counter, 0)
        self.assertFalse(idle_scheduler._tpu_drain_prefill)

    def test_configure_tpu_scheduler_hook_reads_and_clamps_env(self):
        scheduler = _make_scheduler(interval=0, require_mlp_sync=True)
        self.assertIn(
            (
                "sglang.srt.managers.scheduler.Scheduler.__init__",
                configure_tpu_scheduler,
                HookType.AFTER,
            ),
            TPU_HOOKS,
        )

        with patch.dict(os.environ, {"SGLANG_TPU_PREFILL_DRAIN_INTERVAL": "256"}):
            configure_tpu_scheduler(None, scheduler)
        self.assertEqual(scheduler._tpu_prefill_drain_interval, 256)

        with patch.dict(os.environ, {"SGLANG_TPU_PREFILL_DRAIN_INTERVAL": "-5"}):
            configure_tpu_scheduler(None, scheduler)
        self.assertEqual(scheduler._tpu_prefill_drain_interval, 0)


if __name__ == "__main__":
    unittest.main()
