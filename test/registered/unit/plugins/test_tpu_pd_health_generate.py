"""P/D /health_generate from held requests producing output, TPU plugin.

Must not come back: an engine whose requests produce nothing for the whole
window answering 200. Its mirror: an idle engine, or one whose requests do
produce output, answering 503 from this rule.

Usage:
    python3 -m pytest test/registered/unit/plugins/test_tpu_pd_health_generate.py -v
"""

import asyncio
import os
import tempfile
import typing
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
from fastapi import Request

from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.entrypoints import http_server
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import patches
from sglang_tpu.disaggregation import health_generate as hgm
from sglang_tpu.disaggregation import health_generate_http as hh
from sglang_tpu.disaggregation import health_generate_scheduler as hs
from sglang_tpu.disaggregation.health_generate import (
    PDHealthConfig,
    PDHealthSlot,
    PDHealthSnapshot,
    PDHealthState,
    PDProgressHealth,
    read_slots,
    slot_path,
    write_slot,
)
from sglang_tpu.envs import envs as tpu_envs

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

# Window T = 100 s after the first progress event, 250 s before it.
CFG = PDHealthConfig(window_s=100.0, cold_window_s=250.0)


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _warm_tracker(role="decode"):
    """A tracker that has produced once, so the normal window applies."""
    clock = _Clock()
    tracker = PDProgressHealth(config=CFG, clock=clock, role=role)
    tracker.observe(1)
    tracker.record_progress()
    tracker.observe(0)
    clock.advance(5000.0)  # long idle: the last progress is far in the past
    return tracker, clock


class TestRule(CustomTestCase):
    def test_idle_engine_is_ready_whatever_the_progress_age(self):
        tracker, clock = _warm_tracker()
        self.assertEqual(tracker.observe(0), PDHealthState.READY)
        clock.advance(24 * 3600.0)
        self.assertEqual(tracker.observe(0), PDHealthState.READY)
        self.assertEqual(tracker.snapshot().reason, "ready")

    def test_first_request_after_idle_gets_the_whole_window(self):
        """The window counts from when requests appeared, not from the last output."""
        tracker, clock = _warm_tracker()
        tracker.observe(1)
        clock.advance(CFG.window_s - 1)
        self.assertEqual(tracker.observe(1), PDHealthState.READY)
        clock.advance(1)
        self.assertEqual(tracker.observe(1), PDHealthState.UNHEALTHY)

    def test_progress_inside_the_window_keeps_ready(self):
        tracker, clock = _warm_tracker()
        tracker.observe(2)
        clock.advance(90)
        tracker.record_progress()  # one request produced something
        clock.advance(90)
        self.assertEqual(tracker.observe(2), PDHealthState.READY, "progress 90 s ago")
        clock.advance(10)
        self.assertEqual(tracker.observe(2), PDHealthState.UNHEALTHY, "100 s silent")

    def test_count_dropping_to_zero_restarts_the_clock(self):
        """Fast-failing requests with gaps between them never trip the rule."""
        tracker, clock = _warm_tracker()
        for _ in range(5):
            tracker.observe(1)
            clock.advance(60)
            self.assertEqual(tracker.observe(0), PDHealthState.READY)
            clock.advance(1)
        tracker.observe(1)
        clock.advance(CFG.window_s)
        self.assertEqual(tracker.observe(1), PDHealthState.UNHEALTHY)

    def test_continuous_demand_with_churn_trips(self):
        """Requests come and go but the engine is never empty: that is demand."""
        tracker, clock = _warm_tracker()
        for count in (3, 2, 5, 1):
            tracker.observe(count)
            clock.advance(CFG.window_s / 4)
        self.assertEqual(tracker.observe(1), PDHealthState.UNHEALTHY)

    def test_failures_at_a_timeout_shorter_than_the_window_keep_demand_alive(self):
        """Live 2P/2D run: every transfer into the decode failed at the 60 s transport
        timeout, so the queue drained before the window closed and the decode stayed
        ready for the whole run. An engine-concluded failure must count as demand."""
        tracker, clock = _warm_tracker()
        for _ in range(3):  # requests arrive, hang 60 s, fail; queue empty in between
            tracker.observe(2)
            clock.advance(30)
            tracker.observe(2)
            clock.advance(30)
            tracker.record_failure()
            tracker.observe(0)
            clock.advance(2)
        self.assertEqual(tracker.state, PDHealthState.UNHEALTHY)
        self.assertIn("last failed", tracker.snapshot().reason)

    def test_client_cancels_do_not_count_as_demand(self):
        """Requests cancelled by clients leave the count and record nothing."""
        tracker, clock = _warm_tracker()
        for _ in range(5):
            tracker.observe(3)
            clock.advance(50)
            tracker.observe(0)  # all three cancelled; no record_failure
            clock.advance(1)
        self.assertEqual(tracker.state, PDHealthState.READY)

    def test_failures_age_out_so_recovery_still_needs_no_state(self):
        tracker, clock = _warm_tracker()
        tracker.observe(1)
        clock.advance(CFG.window_s)
        tracker.record_failure()
        tracker.observe(0)
        self.assertEqual(
            tracker.state, PDHealthState.UNHEALTHY, "demand kept by the failure"
        )
        clock.advance(CFG.window_s - 1)
        self.assertEqual(tracker.observe(0), PDHealthState.UNHEALTHY)
        clock.advance(1)
        self.assertEqual(tracker.observe(0), PDHealthState.READY, "failure aged out")
        self.assertIsNone(tracker.snapshot().demand_age_s)

    def test_window_boundary_is_inclusive_on_both_halves(self):
        """Exactly T of demand and exactly T since progress trips; one tick less does not."""
        tracker, clock = _warm_tracker()
        tracker.record_progress()  # progress now; demand starts now too
        tracker.observe(1)
        clock.advance(CFG.window_s - 0.5)
        self.assertEqual(tracker.observe(1), PDHealthState.READY)
        clock.advance(0.5)
        self.assertEqual(tracker.observe(1), PDHealthState.UNHEALTHY)

    def test_progress_alone_never_changes_demand(self):
        """A streaming decode produces every few ms; demand still dates from arrival."""
        tracker, clock = _warm_tracker()
        tracker.observe(1)
        for _ in range(50):
            clock.advance(1)
            tracker.record_progress()
        self.assertEqual(tracker.snapshot().demand_age_s, 50.0)
        self.assertEqual(tracker.state, PDHealthState.READY)

    def test_a_lone_failure_never_trips_by_itself(self):
        """One engine abort with nothing else in flight is noise: it starts demand,
        but it ages out at the very moment the window would close."""
        tracker, clock = _warm_tracker()
        tracker.record_failure()
        self.assertEqual(tracker.snapshot().demand_age_s, 0.0)
        clock.advance(CFG.window_s)
        self.assertEqual(tracker.observe(0), PDHealthState.READY)
        self.assertIsNone(tracker.snapshot().demand_age_s)

    def test_failures_with_a_success_in_between_never_trip(self):
        """A burst of engine failures around successes is noise, not a stall."""
        tracker, clock = _warm_tracker()
        for _ in range(10):
            tracker.observe(2)
            tracker.record_failure()
            clock.advance(40)
            tracker.record_progress()
            clock.advance(40)
        self.assertEqual(tracker.observe(2), PDHealthState.READY)

    def test_cold_window_also_governs_how_long_a_failure_keeps_demand(self):
        """Before the first progress, a failure bridges gaps for the cold window, not T."""
        clock = _Clock()
        tracker = PDProgressHealth(config=CFG, clock=clock)
        tracker.observe(1)
        tracker.record_failure()
        tracker.observe(0)  # queue empty right after the failure
        clock.advance(CFG.window_s + 1)
        self.assertEqual(
            tracker.observe(0), PDHealthState.READY, "cold window still open"
        )
        self.assertEqual(
            tracker.snapshot().demand_age_s, CFG.window_s + 1, "demand bridged"
        )
        tracker.observe(1)  # a new request arrives before the failure aged out
        clock.advance(CFG.cold_window_s - CFG.window_s - 1)
        self.assertEqual(
            tracker.observe(1), PDHealthState.UNHEALTHY, "demand continuous since t0"
        )

    def test_negative_count_is_treated_as_empty(self):
        """A bookkeeping slip must never manufacture demand."""
        tracker, clock = _warm_tracker()
        tracker.observe(-1)
        self.assertIsNone(tracker.snapshot().demand_age_s)
        self.assertEqual(tracker.snapshot().requests_present, -1)

    def test_role_is_carried_in_the_record(self):
        prefill, _ = _warm_tracker("prefill")
        self.assertEqual(prefill.snapshot().role, "prefill")
        self.assertEqual(
            PDHealthSnapshot.from_json(prefill.health_json()).role, "prefill"
        )

    def test_verdict_ages_without_any_event(self):
        """Live 2P/2D run (HTTP-side variant): the decode stayed 503 for minutes
        after its last abort had aged out, because nothing called observe() while
        no requests flowed. A read must re-judge on the clock alone."""
        tracker, clock = _warm_tracker()
        tracker.observe(1)
        clock.advance(CFG.window_s)
        tracker.record_failure()
        tracker.observe(0)
        self.assertEqual(tracker.health_state(), "unhealthy")
        clock.advance(CFG.window_s)  # the failure ages out; no observe() call
        self.assertEqual(tracker.health_state(), "ready")
        self.assertIsNone(tracker.snapshot().demand_age_s)
        self.assertEqual(
            tracker.version, 2, "the read flipped the verdict exactly once"
        )

    def test_cold_window_applies_until_the_first_progress(self):
        clock = _Clock()
        tracker = PDProgressHealth(config=CFG, clock=clock)
        tracker.observe(1)
        clock.advance(CFG.window_s)
        self.assertEqual(
            tracker.observe(1), PDHealthState.READY, "cold compile allowed"
        )
        self.assertEqual(tracker.snapshot().window_s, CFG.cold_window_s)
        clock.advance(CFG.cold_window_s - CFG.window_s)
        self.assertEqual(tracker.observe(1), PDHealthState.UNHEALTHY)
        self.assertIn("since start", tracker.snapshot().reason)
        tracker.record_progress()
        self.assertEqual(tracker.state, PDHealthState.READY)
        self.assertEqual(tracker.snapshot().window_s, CFG.window_s)
        clock.advance(CFG.window_s)
        self.assertEqual(
            tracker.observe(1), PDHealthState.UNHEALTHY, "normal window now"
        )

    def test_recovery_needs_no_state(self):
        """Out of service, the held requests leave; the engine is ready again at once
        and the next request gets a fresh window: the issue's probation for free."""
        tracker, clock = _warm_tracker()
        tracker.observe(4)
        clock.advance(CFG.window_s)
        self.assertEqual(tracker.observe(4), PDHealthState.UNHEALTHY)
        self.assertEqual(tracker.version, 1)
        self.assertEqual(tracker.observe(0), PDHealthState.READY)
        self.assertEqual(tracker.version, 2)
        tracker.observe(1)
        clock.advance(CFG.window_s - 1)
        self.assertEqual(tracker.observe(1), PDHealthState.READY)
        clock.advance(1)
        self.assertEqual(tracker.observe(1), PDHealthState.UNHEALTHY)
        self.assertEqual(tracker.version, 3, "version counts state changes only")

    def test_snapshot_reports_the_facts_behind_the_verdict(self):
        tracker, clock = _warm_tracker()
        tracker.observe(2)
        clock.advance(CFG.window_s + 5)
        tracker.observe(2)
        snap = tracker.snapshot()
        self.assertFalse(snap.ready)
        self.assertEqual(snap.requests_present, 2)
        self.assertEqual(snap.demand_age_s, 105.0)
        self.assertEqual(snap.last_progress_age_s, 5105.0)
        self.assertEqual(
            snap.reason,
            "stalled: 2 requests held, demand for 105s, nothing produced for 5105s (window 100s)",
        )

    def test_invalid_config_is_rejected_and_env_is_read(self):
        for bad in (dict(window_s=0.0), dict(window_s=10.0, cold_window_s=5.0)):
            with self.assertRaises(ValueError, msg=bad):
                PDProgressHealth(config=PDHealthConfig(**bad), clock=_Clock())
        with (
            tpu_envs.SGLANG_TPU_PD_HEALTH_WINDOW_S.override(12.5),
            tpu_envs.SGLANG_TPU_PD_HEALTH_COLD_WINDOW_S.override(99.0),
        ):
            self.assertEqual(
                PDHealthConfig.from_env(),
                PDHealthConfig(window_s=12.5, cold_window_s=99.0),
            )


# ---------------------------------------------------------------------------
# The slot files
# ---------------------------------------------------------------------------


class _SlotDir:
    """SLOT_ROOT pointed at a fresh temporary directory."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._patch = patch.object(hgm, "SLOT_ROOT", self._tmp.name)
        self._patch.start()
        return self._tmp.name

    def __exit__(self, *exc):
        self._patch.stop()
        self._tmp.cleanup()


def _slot(**kw):
    base = dict(pid=os.getpid(), written_at=1.0)
    base.update(kw)
    return PDHealthSlot(**base)


class TestSlots(CustomTestCase):
    def test_round_trip_and_ordering(self):
        with _SlotDir():
            write_slot(slot_path(30000, dp_rank=0, tp_rank=1), _slot(tp_rank=1))
            write_slot(slot_path(30000, dp_rank=0, tp_rank=0), _slot(tp_rank=0))
            self.assertEqual([s.tp_rank for s in read_slots(30000)], [0, 1])
            self.assertEqual(read_slots(30001), [], "another port's directory")

    def test_rewrite_replaces_the_rank_file(self):
        with _SlotDir():
            path = slot_path(30000, dp_rank=0, tp_rank=0)
            write_slot(path, _slot(state="unhealthy"))
            write_slot(path, _slot(state="ready"))
            (slot,) = read_slots(30000)
            self.assertEqual(slot.state, "ready")

    def test_a_dead_writers_slot_is_a_leftover(self):
        with _SlotDir():
            write_slot(slot_path(30000, dp_rank=0, tp_rank=0), _slot(pid=2**22 + 7))
            write_slot(slot_path(30000, dp_rank=0, tp_rank=1), _slot(tp_rank=1))
            self.assertEqual([s.tp_rank for s in read_slots(30000)], [1])

    def test_garbage_is_skipped(self):
        with _SlotDir():
            path = slot_path(30000, dp_rank=0, tp_rank=0)
            os.makedirs(os.path.dirname(path))
            with open(path, "w") as f:
                f.write("{not json")
            self.assertEqual(read_slots(30000), [])


# ---------------------------------------------------------------------------
# The scheduler-side bookkeeping, and the patch that calls it
# ---------------------------------------------------------------------------


def _req(host="10.0.0.1", finished=None):
    return SimpleNamespace(bootstrap_host=host, finished_reason=finished)


def _scheduler(mode, *, waiting=(), running=(), **queues):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.disaggregation_mode = mode
    scheduler.server_args = SimpleNamespace(port=30000)
    scheduler.output_streamer = SimpleNamespace(disaggregation_mode=mode)
    scheduler.waiting_queue = list(waiting)
    scheduler.running_batch = SimpleNamespace(reqs=list(running))
    scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
        queue=list(queues.get("bootstrap", ()))
    )
    scheduler.disagg_prefill_inflight_queue = list(queues.get("inflight", ()))
    scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
        queue=[SimpleNamespace(req=r) for r in queues.get("prealloc", ())],
        retracted_queue=list(queues.get("retracted", ())),
    )
    scheduler.disagg_decode_transfer_queue = SimpleNamespace(
        queue=[SimpleNamespace(req=r) for r in queues.get("transfer", ())]
    )
    return scheduler


_RANKS = SimpleNamespace(dp_rank=None, tp_rank=0)


def _attached(mode=DisaggregationMode.DECODE, **queues):
    """A scheduler double with a tracker attached on an injected clock."""
    scheduler = _scheduler(mode, **queues)
    clock = _Clock()
    state = hs._SchedulerHealth(
        PDProgressHealth(config=CFG, clock=clock, role=mode.value),
        slot_path(30000, dp_rank=0, tp_rank=0),
    )
    hs._BY_SCHEDULER[scheduler] = state
    hs._BY_STREAMER[id(scheduler.output_streamer)] = state
    return scheduler, clock


class TestSchedulerBookkeeping(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._dir = _SlotDir()
        self._dir.__enter__()
        self._ranks = patch.object(hs, "get_parallel", lambda: _RANKS)
        self._ranks.start()

    def tearDown(self):
        self._ranks.stop()
        self._dir.__exit__(None, None, None)
        hs._BY_STREAMER.clear()
        super().tearDown()

    def test_attach_only_for_pd_roles_and_only_when_enabled(self):
        for mode, expect in (
            (DisaggregationMode.DECODE, True),
            (DisaggregationMode.PREFILL, True),
            (DisaggregationMode.NULL, False),
        ):
            scheduler = _scheduler(mode)
            hs.attach_tracker(scheduler)
            self.assertEqual(hs.state_of(scheduler) is not None, expect, mode)
            if expect:
                self.assertEqual(hs.state_of(scheduler).tracker.role, mode.value)
                (slot,) = read_slots(30000)
                self.assertEqual((slot.state, slot.pid), ("ready", os.getpid()))
        with tpu_envs.SGLANG_TPU_PD_HEALTH_READINESS_ENABLED.override(False):
            scheduler = _scheduler(DisaggregationMode.DECODE)
            hs.attach_tracker(scheduler)
        self.assertIsNone(hs.state_of(scheduler))

    def test_requests_present_counts_every_pd_queue_and_skips_probes(self):
        probe = _req(FAKE_BOOTSTRAP_HOST)
        prefill = _scheduler(
            DisaggregationMode.PREFILL,
            waiting=[_req(), probe],
            running=[_req()],
            bootstrap=[_req(), _req()],
            inflight=[_req(), probe],
        )
        self.assertEqual(hs.requests_present(prefill), 5)
        decode = _scheduler(
            DisaggregationMode.DECODE,
            waiting=[probe],
            running=[_req()],
            prealloc=[_req(), _req()],
            transfer=[_req()],
            retracted=[_req(), probe],
        )
        self.assertEqual(hs.requests_present(decode), 5)

    def test_decode_streams_are_progress_and_5xx_aborts_are_failures(self):
        scheduler, clock = _attached()
        tracker = hs.state_of(scheduler).tracker
        streamer = scheduler.output_streamer
        hs.track_streamed(streamer, [_req(), _req(FAKE_BOOTSTRAP_HOST)])  # tokens
        self.assertEqual(tracker.snapshot().last_progress_age_s, 0.0)
        self.assertIsNone(tracker.snapshot().last_failure_age_s)
        clock.advance(10)
        hs.track_streamed(
            streamer,
            [_req(finished=FINISH_ABORT("x", HTTPStatus.INTERNAL_SERVER_ERROR))],
        )
        self.assertEqual(tracker.snapshot().last_failure_age_s, 0.0)
        self.assertEqual(tracker.snapshot().last_progress_age_s, 10.0)

    def test_client_cancels_and_rejections_count_for_nothing(self):
        scheduler, clock = _attached()
        tracker = hs.state_of(scheduler).tracker
        hs.track_streamed(
            scheduler.output_streamer,
            [
                _req(finished=FINISH_ABORT()),  # client cancel
                _req(finished=FINISH_ABORT("too long", HTTPStatus.BAD_REQUEST)),
                _req(FAKE_BOOTSTRAP_HOST, FINISH_ABORT("probe", 500)),
            ],
        )
        snap = tracker.snapshot()
        self.assertIsNone(snap.last_failure_age_s)
        self.assertIsNone(snap.last_progress_age_s)

    def test_prefill_counts_only_final_outputs_as_progress(self):
        scheduler, clock = _attached(DisaggregationMode.PREFILL)
        tracker = hs.state_of(scheduler).tracker
        hs.track_streamed(scheduler.output_streamer, [_req()])  # mid-prefill chunk
        self.assertIsNone(tracker.snapshot().last_progress_age_s)
        hs.track_streamed(scheduler.output_streamer, [_req(finished=FINISH_LENGTH(0))])
        self.assertEqual(tracker.snapshot().last_progress_age_s, 0.0)

    def test_streams_from_an_unknown_streamer_are_ignored(self):
        hs.track_streamed(SimpleNamespace(disaggregation_mode=None), [_req()])

    def test_observe_publishes_exactly_once_per_verdict_change(self):
        scheduler, clock = _attached(DisaggregationMode.DECODE, running=[_req()])
        state = hs.state_of(scheduler)
        state.tracker.record_progress()
        with patch.object(hs, "_publish", wraps=hs._publish) as publish:
            hs.observe_and_publish(scheduler)  # ready, first publish
            hs.observe_and_publish(scheduler)  # unchanged, within the heartbeat
            clock.advance(CFG.window_s)
            hs.observe_and_publish(scheduler)  # flips to unhealthy
            scheduler.running_batch.reqs.clear()
            hs.observe_and_publish(scheduler)  # back to ready
            hs.observe_and_publish(scheduler)  # idle, nothing due
        self.assertEqual(publish.call_count, 3)
        (slot,) = read_slots(30000)
        self.assertEqual((slot.state, slot.requests_present), ("ready", 0))

    def test_held_count_change_republishes_without_a_verdict_change(self):
        """While unhealthy on a recent failure, the queue draining to zero must
        reach the slot, or the HTTP side reads a frozen count as a stopped loop."""
        scheduler, clock = _attached(DisaggregationMode.DECODE, running=[_req()])
        tracker = hs.state_of(scheduler).tracker
        tracker.record_progress()
        hs.observe_and_publish(scheduler)
        clock.advance(CFG.window_s)
        hs.observe_and_publish(scheduler)  # unhealthy, 1 held
        tracker.record_failure()
        scheduler.running_batch.reqs.clear()
        hs.observe_and_publish(scheduler)  # still unhealthy, 0 held
        (slot,) = read_slots(30000)
        self.assertEqual((slot.state, slot.requests_present), ("unhealthy", 0))

    def test_slot_is_refreshed_while_work_is_held_even_unchanged(self):
        scheduler, clock = _attached(DisaggregationMode.DECODE, running=[_req()])
        state = hs.state_of(scheduler)
        with patch.object(hs, "_publish", wraps=hs._publish) as publish:
            hs.observe_and_publish(scheduler)  # first slot
            state.published_ts -= hs.PD_HEALTH_REPUBLISH_S
            hs.observe_and_publish(scheduler)  # heartbeat, nothing changed
            scheduler.running_batch.reqs.clear()
            hs.observe_and_publish(scheduler)  # held count changed: publish...
            state.published_ts -= hs.PD_HEALTH_REPUBLISH_S
            hs.observe_and_publish(scheduler)  # ...then no heartbeat while idle
        self.assertEqual(publish.call_count, 3)

    def test_slot_carries_the_staleness_facts(self):
        scheduler, clock = _attached(DisaggregationMode.PREFILL, inflight=[_req()] * 3)
        hs.observe_and_publish(scheduler)
        (slot,) = read_slots(30000)
        self.assertEqual(
            (slot.requests_present, slot.stale_after_s), (3, CFG.cold_window_s)
        )
        self.assertEqual(PDHealthSnapshot.from_json(slot.record).role, "prefill")
        self.assertGreater(slot.written_at, 0)

    def test_a_failed_slot_write_is_logged_once_and_never_raises(self):
        scheduler, clock = _attached(DisaggregationMode.DECODE, running=[_req()])
        state = hs.state_of(scheduler)
        state.path = os.path.join(os.devnull, "x", "y.json")
        with patch.object(hs.logger, "warning") as warning:
            hs.observe_and_publish(scheduler)
            state.published_ts -= hs.PD_HEALTH_REPUBLISH_S
            hs.observe_and_publish(scheduler)
        self.assertEqual(warning.call_count, 1)

    def test_bookkeeping_is_a_no_op_on_a_scheduler_without_a_tracker(self):
        scheduler = _scheduler(DisaggregationMode.NULL, running=[_req()])
        hs.observe_and_publish(scheduler)
        self.assertEqual(read_slots(30000), [])


class _StubScheduler:
    def __init__(self):
        self.calls = []

    def init_output_streamer(self):
        self.calls.append("init_output_streamer")
        return "streamer"

    def process_decode_queue(self):
        self.calls.append("decode_queue")

    def process_disagg_prefill_inflight_queue(self, *, extra=None):
        self.calls.append(("inflight", extra))
        return "polled"


class _StubStreamer:
    def __init__(self):
        self.calls = []

    def stream_output(self, reqs, return_logprob, skip_req=None):
        self.calls.append((list(reqs), return_logprob, skip_req))
        return "streamed"


class TestPatch(CustomTestCase):
    def tearDown(self):
        hs._BY_STREAMER.clear()
        super().tearDown()

    def test_the_core_methods_exist(self):
        for cls, name in (
            (Scheduler, "init_output_streamer"),
            (SchedulerDisaggregationDecodeMixin, "process_decode_queue"),
            (
                SchedulerDisaggregationPrefillMixin,
                "process_disagg_prefill_inflight_queue",
            ),
            (SchedulerOutputStreamer, "stream_output"),
            (TokenizerManager, "init_disaggregation"),
        ):
            self.assertTrue(callable(getattr(cls, name)), name)

    def test_scheduler_patch_wraps_once_and_keeps_the_originals(self):
        originals = (
            Scheduler.init_output_streamer,
            SchedulerDisaggregationDecodeMixin.process_decode_queue,
            SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue,
            SchedulerOutputStreamer.stream_output,
        )
        flag = patches._scheduler_pd_health_patched
        try:
            patches._scheduler_pd_health_patched = False
            patches._patch_scheduler_pd_health_generate()
            once = SchedulerOutputStreamer.stream_output
            self.assertIs(once.__wrapped__, originals[3])
            patches._patch_scheduler_pd_health_generate()
            self.assertIs(SchedulerOutputStreamer.stream_output, once)
        finally:
            patches._scheduler_pd_health_patched = flag
            Scheduler.init_output_streamer = originals[0]
            SchedulerDisaggregationDecodeMixin.process_decode_queue = originals[1]
            SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue = originals[
                2
            ]
            SchedulerOutputStreamer.stream_output = originals[3]

    def test_http_patches_apply_only_in_pd_mode(self):
        with patch.object(
            patches, "_patch_tokenizer_manager_pd_health_generate"
        ) as inst:
            patches.apply_tpu_http_patches(disaggregation_mode="null")
            inst.assert_not_called()
            patches.apply_tpu_http_patches(disaggregation_mode="prefill")
            inst.assert_called_once()

    def test_scheduler_wrappers_call_the_bookkeeping_around_the_originals(self):
        sched_cls = type("S", (_StubScheduler,), {})
        streamer_cls = type("O", (_StubStreamer,), {})
        patches._wrap_scheduler_for_pd_health(
            scheduler_cls=sched_cls,
            decode_cls=sched_cls,
            prefill_cls=sched_cls,
            streamer_cls=streamer_cls,
        )
        scheduler, streamer = sched_cls(), streamer_cls()
        with (
            patch.object(hs, "attach_tracker") as attach,
            patch.object(hs, "observe_and_publish") as observe,
            patch.object(hs, "track_streamed") as track,
        ):
            self.assertEqual(scheduler.init_output_streamer(), "streamer")
            attach.assert_called_once_with(scheduler)
            scheduler.process_decode_queue()
            self.assertEqual(
                scheduler.process_disagg_prefill_inflight_queue(extra=1), "polled"
            )
            self.assertEqual(observe.call_count, 2)
            self.assertEqual(streamer.stream_output(["r"], True), "streamed")
            track.assert_called_once_with(streamer, ["r"])
        self.assertEqual(
            scheduler.calls, ["init_output_streamer", "decode_queue", ("inflight", 1)]
        )
        self.assertEqual(streamer.calls, [(["r"], True, None)])

    def test_tokenizer_manager_wrapper_attaches_after_the_original(self):
        class Stub:
            def __init__(self):
                self.calls = []

            def init_disaggregation(self, *, start_pd_bootstrap_service=True):
                self.calls.append(start_pd_bootstrap_service)
                return "init-result"

        patches._wrap_tokenizer_manager_for_pd_health(Stub)
        tm = Stub()
        with patch.object(hh, "attach_slots") as attach:
            self.assertEqual(
                tm.init_disaggregation(start_pd_bootstrap_service=False), "init-result"
            )
        attach.assert_called_once_with(tm)
        self.assertEqual(tm.calls, [False])


# ---------------------------------------------------------------------------
# The /health_generate route
# ---------------------------------------------------------------------------


def _tm(role=DisaggregationMode.DECODE, port=30000):
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.disaggregation_mode = role
    tm.server_args = SimpleNamespace(port=port)
    return tm


def _write_rank(tp_rank, tracker=None, *, age_s=0.0, state=None):
    """A rank's slot as the scheduler writes it; ``age_s`` backdates the write."""
    import time as _time

    slot = PDHealthSlot(
        tp_rank=tp_rank, pid=os.getpid(), written_at=_time.time() - age_s
    )
    if tracker is not None:
        slot.state = tracker.health_state()
        slot.record = tracker.health_json()
        slot.requests_present = tracker.requests_present()
        slot.stale_after_s = tracker.stale_after_s()
    if state is not None:
        slot.state = state
    write_slot(slot_path(30000, dp_rank=0, tp_rank=tp_rank), slot)


def _stalled_tracker(role="decode"):
    clock = _Clock()
    tracker = PDProgressHealth(config=CFG, clock=clock, role=role)
    tracker.observe(3)
    clock.advance(CFG.cold_window_s)
    tracker.observe(3)
    return tracker


def _busy_tracker():
    tracker = PDProgressHealth(config=CFG, clock=_Clock(), role="decode")
    tracker.record_progress()
    tracker.observe(2)
    return tracker


class TestRoute(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._dir = _SlotDir()
        self._dir.__enter__()
        self._routes = list(http_server.app.router.routes)
        self._orig = dict(hh._ORIGINAL_HANDLER)
        self._prior_state = http_server.get_global_state()
        self.probe_calls = []

        async def probe(request):
            self.probe_calls.append(request.url.path)
            return SimpleNamespace(status_code=200)

        hh._ORIGINAL_HANDLER["handler"] = probe

    def tearDown(self):
        http_server.app.router.routes[:] = self._routes
        hh._ORIGINAL_HANDLER.clear()
        hh._ORIGINAL_HANDLER.update(self._orig)
        http_server._global_state = self._prior_state
        self._dir.__exit__(None, None, None)
        super().tearDown()

    def _route(self, path):
        from fastapi.routing import APIRoute

        return [
            r
            for r in http_server.app.router.routes
            if isinstance(r, APIRoute) and r.path == path
        ]

    def _call(self, tm):
        http_server.set_global_state(SimpleNamespace(tokenizer_manager=tm))
        request = SimpleNamespace(url=SimpleNamespace(path=hh.HEALTH_GENERATE_PATH))
        return asyncio.run(hh.health_generate_with_slots(request))

    def _attached(self, role=DisaggregationMode.DECODE):
        tm = _tm(role)
        hh._PORT[tm] = 30000
        return tm

    def test_attach_only_for_pd_roles_and_only_when_enabled(self):
        for role, expect in (
            (DisaggregationMode.DECODE, True),
            (DisaggregationMode.PREFILL, True),
            (DisaggregationMode.NULL, False),
        ):
            tm = _tm(role)
            with patch.object(hh, "install_health_generate_route", lambda: True):
                hh.attach_slots(tm)
            self.assertEqual(hh.port_of(tm), 30000 if expect else None, role)
        tm = _tm()
        with (
            tpu_envs.SGLANG_TPU_PD_HEALTH_READINESS_ENABLED.override(False),
            patch.object(hh, "install_health_generate_route", lambda: True),
        ):
            hh.attach_slots(tm)
        self.assertIsNone(hh.port_of(tm))

    def test_install_swaps_only_health_generate_and_is_idempotent(self):
        hh._ORIGINAL_HANDLER.clear()
        self.assertTrue(hh.install_health_generate_route())
        self.assertFalse(hh.install_health_generate_route())
        (route,) = self._route(hh.HEALTH_GENERATE_PATH)
        self.assertIs(route.endpoint, hh.health_generate_with_slots)
        self.assertEqual(sorted(route.methods), ["GET"])
        self.assertIs(typing.get_type_hints(route.endpoint)["request"], Request)
        (h,) = self._route("/health")
        self.assertIs(h.endpoint, http_server.health_generate)
        self.assertIs(hh._ORIGINAL_HANDLER["handler"], http_server.health_generate)

    def test_stalled_rank_returns_503_with_the_facts_before_probing(self):
        _write_rank(0, _stalled_tracker())
        response = self._call(self._attached())
        self.assertEqual(response.status_code, 503)
        body = msgspec.json.decode(response.body)
        self.assertEqual((body["role"], body["state"]), ("decode", "unhealthy"))
        self.assertIn("stalled: 3 requests held, demand for 250s", body["reason"])
        self.assertEqual(body["pd_health"][0]["requests_present"], 3)
        self.assertEqual(self.probe_calls, [])

    def test_ready_ranks_fall_through_to_the_probe(self):
        _write_rank(0, _busy_tracker())
        _write_rank(1)
        self.assertEqual(self._call(self._attached()).status_code, 200)
        self.assertEqual(self.probe_calls, [hh.HEALTH_GENERATE_PATH])

    def test_no_slots_and_no_tracker_fall_through_to_the_probe(self):
        self.assertEqual(self._call(self._attached()).status_code, 200)
        self.assertEqual(self._call(_tm(DisaggregationMode.NULL)).status_code, 200)
        self.assertEqual(len(self.probe_calls), 2)

    def test_frozen_slot_with_work_held_is_unhealthy(self):
        tracker = _busy_tracker()
        _write_rank(0, tracker, age_s=CFG.window_s)
        body = msgspec.json.decode(self._call(self._attached()).body)
        self.assertEqual(body["state"], "unhealthy")
        self.assertIn(
            "scheduler stopped judging: slot 100s old with 2 requests held",
            body["reason"],
        )
        self.assertEqual(body["pd_health"][0]["slot_age_s"], 100.0)
        self.assertEqual(self.probe_calls, [])

    def test_fresh_slot_with_work_held_is_ready(self):
        _write_rank(0, _busy_tracker(), age_s=CFG.window_s - 1)
        self.assertEqual(self._call(self._attached()).status_code, 200)

    def test_frozen_slot_without_work_is_ready(self):
        tracker = PDProgressHealth(config=CFG, clock=_Clock(), role="decode")
        _write_rank(0, tracker, age_s=10 * CFG.cold_window_s)
        self.assertEqual(self._call(self._attached()).status_code, 200)

    def test_a_dead_ranks_slot_is_ignored(self):
        import time as _time

        write_slot(
            slot_path(30000, dp_rank=0, tp_rank=0),
            PDHealthSlot(pid=2**22 + 7, written_at=_time.time(), state="unhealthy"),
        )
        self.assertEqual(self._call(self._attached()).status_code, 200)

    def test_one_unready_rank_among_many_is_enough(self):
        _write_rank(0, _busy_tracker())
        _write_rank(1, _stalled_tracker())
        _write_rank(2)
        body = msgspec.json.decode(self._call(self._attached()).body)
        self.assertEqual(len(body["pd_health"]), 1)

    def test_prefill_is_gated_too(self):
        _write_rank(0, _stalled_tracker("prefill"))
        body = msgspec.json.decode(
            self._call(self._attached(DisaggregationMode.PREFILL)).body
        )
        self.assertEqual(body["role"], "prefill")


if __name__ == "__main__":
    unittest.main()
