"""P/D /health_generate from in-flight requests producing output, TPU plugin.

Must not come back: an engine whose requests produce nothing for the whole
window answering 200. Its mirror: an idle engine, or one whose requests do
produce output, answering 503 from this rule.

Usage:
    python3 -m pytest test/registered/unit/plugins/test_tpu_pd_health_generate.py -v
"""

import asyncio
import typing
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
from fastapi import Request

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.entrypoints import http_server
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import ServerStatus, TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import patches
from sglang_tpu.disaggregation import health_generate_http as hg
from sglang_tpu.disaggregation.health_generate import (
    PDHealthConfig,
    PDHealthSnapshot,
    PDHealthState,
    PDPeerHealth,
    PDProgressHealth,
    build_pd_health_tracker,
    build_pd_peer_tracker,
)
from sglang_tpu.envs import envs as tpu_envs

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

# Window T = 100 s after the first progress event, 250 s before it.
CFG = PDHealthConfig(window_s=100.0, cold_window_s=250.0)
ABORT = {"type": "abort", "message": "Aborted", "status_code": 500}
LENGTH = {"type": "length", "length": 0}


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


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------


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

    def test_window_boundary_is_inclusive_on_both_halves(self):
        """Exactly T outstanding and exactly T since progress trips; one tick less does not."""
        tracker, clock = _warm_tracker()
        tracker.record_progress()  # progress now; the request is outstanding from now too
        tracker.observe(1)
        clock.advance(CFG.window_s - 0.5)
        self.assertEqual(tracker.observe(1), PDHealthState.READY)
        clock.advance(0.5)
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

    def test_progress_alone_never_changes_when_requests_became_outstanding(self):
        """A streaming decode produces every few ms; outstanding still dates from arrival."""
        tracker, clock = _warm_tracker()
        tracker.observe(1)
        for _ in range(50):
            clock.advance(1)
            tracker.record_progress()
        self.assertEqual(tracker.snapshot().requests_age_s, 50.0)
        self.assertEqual(tracker.state, PDHealthState.READY)

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

    def test_continuously_outstanding_requests_with_churn_trip(self):
        """Requests come and go but the engine is never empty: continuously outstanding."""
        tracker, clock = _warm_tracker()
        for count in (3, 2, 5, 1):
            tracker.observe(count)
            clock.advance(CFG.window_s / 4)
        self.assertEqual(tracker.observe(1), PDHealthState.UNHEALTHY)

    def test_failures_at_a_timeout_shorter_than_the_window_keep_requests_outstanding(
        self,
    ):
        """A transport timeout shorter than the window drains the queue; the
        engine's own failures must keep requests outstanding or it never trips."""
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

    def test_a_lone_failure_never_trips_by_itself(self):
        """One engine abort with nothing else in flight is noise: it starts the clock,
        but it ages out at the very moment the window would close."""
        tracker, clock = _warm_tracker()
        tracker.record_failure()
        self.assertEqual(tracker.snapshot().requests_age_s, 0.0)
        clock.advance(CFG.window_s)
        self.assertEqual(tracker.observe(0), PDHealthState.READY)
        self.assertIsNone(tracker.snapshot().requests_age_s)

    def test_client_cancels_do_not_count_as_outstanding(self):
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
            tracker.state, PDHealthState.UNHEALTHY, "kept outstanding by the failure"
        )
        clock.advance(CFG.window_s - 1)
        self.assertEqual(tracker.observe(0), PDHealthState.UNHEALTHY)
        clock.advance(1)
        self.assertEqual(tracker.observe(0), PDHealthState.READY, "failure aged out")
        self.assertIsNone(tracker.snapshot().requests_age_s)

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

    def test_verdict_ages_without_any_event(self):
        """Once the last failure aged out, a read must be ready with no new event."""
        tracker, clock = _warm_tracker()
        tracker.observe(1)
        clock.advance(CFG.window_s)
        tracker.record_failure()
        tracker.observe(0)
        self.assertEqual(tracker.health_state(), "unhealthy")
        clock.advance(CFG.window_s)  # the failure ages out; no observe() call
        self.assertEqual(tracker.health_state(), "ready")
        self.assertIsNone(tracker.snapshot().requests_age_s)
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

    def test_cold_window_also_governs_how_long_a_failure_keeps_requests_outstanding(
        self,
    ):
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
            tracker.snapshot().requests_age_s,
            CFG.window_s + 1,
            "bridged by the failure",
        )
        tracker.observe(1)  # a new request arrives before the failure aged out
        clock.advance(CFG.cold_window_s - CFG.window_s - 1)
        self.assertEqual(
            tracker.observe(1), PDHealthState.UNHEALTHY, "outstanding since t0"
        )

    def test_recovery_needs_no_state(self):
        """Out of service, the in-flight requests leave; the engine is ready again at
        once and the next request gets a fresh window: the issue's probation for free."""
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
        self.assertEqual(snap.requests_age_s, 105.0)
        self.assertEqual(snap.last_progress_age_s, 5105.0)
        self.assertIsNone(snap.last_failure_age_s)
        self.assertEqual(
            snap.reason,
            "stalled: 2 requests held, outstanding for 105s, nothing produced for 5105s (window 100s)",
        )

    def test_negative_count_is_treated_as_empty(self):
        """A bookkeeping slip must never manufacture outstanding requests."""
        tracker, clock = _warm_tracker()
        tracker.observe(-1)
        self.assertIsNone(tracker.snapshot().requests_age_s)
        self.assertEqual(tracker.snapshot().requests_present, -1)

    def test_wire_strings_match_the_snapshot(self):
        """health_state/health_json are what the gate reads; the gate keys on the state
        string alone, so it must be the snapshot's state."""
        tracker, clock = _warm_tracker()
        self.assertEqual(tracker.health_state(), "ready")
        tracker.observe(1)
        clock.advance(CFG.window_s)
        tracker.observe(1)
        self.assertEqual(tracker.health_state(), "unhealthy")
        self.assertEqual(
            PDHealthSnapshot.from_json(tracker.health_json()), tracker.snapshot()
        )

    def test_role_is_carried_in_the_record(self):
        prefill, _ = _warm_tracker("prefill")
        self.assertEqual(prefill.snapshot().role, "prefill")
        self.assertEqual(
            PDHealthSnapshot.from_json(prefill.health_json()).role, "prefill"
        )

    def test_invalid_config_is_rejected_and_env_is_read(self):
        for bad in (
            dict(window_s=0.0),
            dict(window_s=-1.0),
            dict(window_s=10.0, cold_window_s=5.0),
        ):
            with self.assertRaises(ValueError, msg=bad):
                PDProgressHealth(config=PDHealthConfig(**bad), clock=_Clock())
        PDProgressHealth(config=PDHealthConfig(window_s=10.0, cold_window_s=10.0))
        with (
            tpu_envs.SGLANG_TPU_PD_HEALTH_WINDOW_S.override(12.5),
            tpu_envs.SGLANG_TPU_PD_HEALTH_COLD_WINDOW_S.override(99.0),
        ):
            self.assertEqual(
                PDHealthConfig.from_env(),
                PDHealthConfig(window_s=12.5, cold_window_s=99.0),
            )


# ---------------------------------------------------------------------------
# Building the tracker
# ---------------------------------------------------------------------------


class TestBuild(CustomTestCase):
    def test_kill_switch_returns_no_tracker(self):
        with tpu_envs.SGLANG_TPU_PD_HEALTH_READINESS_ENABLED.override(False):
            self.assertIsNone(build_pd_health_tracker("decode"))

    def test_each_call_builds_an_independent_tracker_with_the_role(self):
        """Two tokenizer workers must not share a clock."""
        with tpu_envs.SGLANG_TPU_PD_HEALTH_WINDOW_S.override(7.0):
            a = build_pd_health_tracker("prefill")
        b = build_pd_health_tracker("decode")
        self.assertEqual((a.role, a.config.window_s), ("prefill", 7.0))
        a.observe(3)
        self.assertEqual(b.snapshot().requests_present, 0)


# ---------------------------------------------------------------------------
# The bookkeeping on the tokenizer manager, and the patch that calls it
# ---------------------------------------------------------------------------


def _tm(role=DisaggregationMode.DECODE):
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.disaggregation_mode = role
    tm.rid_to_state = {}
    tm.gracefully_exit = False
    tm.server_status = ServerStatus.Up
    return tm


def _attached(role=DisaggregationMode.DECODE):
    """A manager with a tracker attached the way attach_tracker does it, on an
    injected clock; the route swap is not exercised here."""
    tm = _tm(role)
    clock = _Clock()
    hg._STATE[tm] = hg._HttpHealth(
        PDProgressHealth(config=CFG, clock=clock, role=role.value)
    )
    return tm, clock


def _state(peer=None):
    """A ReqState double; ``peer`` is the prefill the router stamped on the request."""
    obj = None
    if peer is not None:
        host, _, port = peer.partition(":")
        obj = GenerateReqInput(text="x", bootstrap_host=host, bootstrap_port=int(port))
    return SimpleNamespace(finished=False, obj=obj)


def _outputs(*pairs):
    return SimpleNamespace(
        rids=[r for r, _ in pairs], finished_reasons=[f for _, f in pairs]
    )


class _StubManager:
    """The four core methods the patch wraps, recording their calls."""

    def __init__(self, role=DisaggregationMode.DECODE):
        self.disaggregation_mode = DisaggregationMode.NULL
        self._role = role
        self.rid_to_state = {}
        self.calls = []

    def init_disaggregation(self, *, start_pd_bootstrap_service=True):
        self.disaggregation_mode = self._role
        self.calls.append(("init", start_pd_bootstrap_service))
        return "init-result"

    def _mark_state_dispatched(self, rid):
        self.calls.append(("dispatched", rid))

    async def _handle_batch_output(self, recv_obj):
        self.calls.append(("output", hg.state_of(self).tracker.snapshot()))

    def abort_request(self, rid="", abort_all=False):
        self.calls.append(("abort", rid, abort_all))
        return "abort-result"


class TestPatch(CustomTestCase):
    def test_the_four_core_methods_exist(self):
        """The patch depends on four core method names; a rename must fail here,
        not at runtime in a stale 200."""
        for name in (
            "init_disaggregation",
            "_mark_state_dispatched",
            "_handle_batch_output",
            "abort_request",
        ):
            self.assertTrue(callable(getattr(TokenizerManager, name)), name)

    def test_patching_the_real_class_is_idempotent(self):
        originals = {
            name: getattr(TokenizerManager, name)
            for name in (
                "init_disaggregation",
                "_mark_state_dispatched",
                "_handle_batch_output",
                "abort_request",
            )
        }
        flag = patches._tokenizer_manager_pd_health_patched
        try:
            patches._tokenizer_manager_pd_health_patched = False
            patches._patch_tokenizer_manager_pd_health_generate()
            once = TokenizerManager._handle_batch_output
            self.assertIsNot(once, originals["_handle_batch_output"])
            self.assertIs(once.__wrapped__, originals["_handle_batch_output"])
            patches._patch_tokenizer_manager_pd_health_generate()
            self.assertIs(TokenizerManager._handle_batch_output, once)
        finally:
            patches._tokenizer_manager_pd_health_patched = flag
            for name, fn in originals.items():
                setattr(TokenizerManager, name, fn)

    def test_http_patches_apply_only_in_pd_mode(self):
        with patch.object(
            patches, "_patch_tokenizer_manager_pd_health_generate"
        ) as installer:
            patches.apply_tpu_http_patches(disaggregation_mode="null")
            installer.assert_not_called()
            patches.apply_tpu_http_patches(disaggregation_mode="decode")
            installer.assert_called_once()

    def _patched_stub(self, role=DisaggregationMode.DECODE):
        cls = type("Stub", (_StubManager,), {})
        patches._wrap_tokenizer_manager_for_pd_health(cls)
        return cls(role)

    def test_init_disaggregation_attaches_after_the_original_ran(self):
        """The role is set by the original, so attaching must come after it."""
        tm = self._patched_stub()
        with patch.object(hg, "install_health_generate_route", lambda: True):
            self.assertEqual(tm.init_disaggregation(), "init-result")
        self.assertEqual(tm.calls, [("init", True)])
        self.assertEqual(hg.state_of(tm).tracker.role, "decode")

    def test_bookkeeping_runs_before_each_original_and_results_pass_through(self):
        tm = self._patched_stub()
        tm.disaggregation_mode = DisaggregationMode.DECODE
        clock = _Clock()
        hg._STATE[tm] = hg._HttpHealth(
            PDProgressHealth(config=CFG, clock=clock, role="decode")
        )
        tm.rid_to_state = {"a": _state(), "b": _state()}
        tm._mark_state_dispatched("a")
        tm._mark_state_dispatched("b")
        clock.advance(7)
        asyncio.run(tm._handle_batch_output(_outputs(("a", None))))
        self.assertEqual(tm.abort_request("b", abort_all=False), "abort-result")
        kinds = [c[0] for c in tm.calls]
        self.assertEqual(kinds, ["dispatched", "dispatched", "output", "abort"])
        seen_by_original = tm.calls[2][1]
        self.assertEqual(
            seen_by_original.last_progress_age_s,
            0.0,
            "progress was recorded before the coroutine body ran",
        )
        self.assertEqual(tm.calls[3], ("abort", "b", False))
        self.assertEqual(hg.state_of(tm).tracker.snapshot().requests_present, 1)


class TestBookkeeping(CustomTestCase):
    def test_attach_only_for_pd_roles_and_only_when_enabled(self):
        for role, expect in (
            (DisaggregationMode.DECODE, True),
            (DisaggregationMode.PREFILL, True),
            (DisaggregationMode.NULL, False),
        ):
            tm = _tm(role)
            with patch.object(hg, "install_health_generate_route", lambda: True):
                hg.attach_tracker(tm)
            self.assertEqual(hg.state_of(tm) is not None, expect, role)
            if expect:
                self.assertEqual(hg.state_of(tm).tracker.role, role.value)
        tm = _tm()
        with (
            tpu_envs.SGLANG_TPU_PD_HEALTH_READINESS_ENABLED.override(False),
            patch.object(hg, "install_health_generate_route", lambda: True),
        ):
            hg.attach_tracker(tm)
        self.assertIsNone(hg.state_of(tm))

    def test_sent_requests_are_outstanding_and_probes_are_not(self):
        tm, clock = _attached()
        tm.rid_to_state = {
            "a": _state(),
            "b": _state(),
            f"{HEALTH_CHECK_RID_PREFIX}_1": _state(),
        }
        hg.track_dispatched(tm, "a")
        hg.track_dispatched(tm, "b")
        hg.track_dispatched(tm, f"{HEALTH_CHECK_RID_PREFIX}_1")
        snap = hg.state_of(tm).tracker.snapshot()
        self.assertEqual((snap.requests_present, snap.requests_age_s), (2, 0.0))

    def test_streaming_chunks_are_progress_and_do_not_end_the_request(self):
        tm, clock = _attached()
        tm.rid_to_state = {"a": _state()}
        hg.track_dispatched(tm, "a")
        clock.advance(10)
        hg.track_outputs(tm, _outputs(("a", None)))  # a token
        snap = hg.state_of(tm).tracker.snapshot()
        self.assertEqual((snap.requests_present, snap.last_progress_age_s), (1, 0.0))
        hg.track_outputs(tm, _outputs(("a", LENGTH)))  # final chunk
        snap = hg.state_of(tm).tracker.snapshot()
        self.assertEqual(snap.requests_present, 0)
        self.assertIsNone(snap.requests_age_s)

    def test_prefill_final_output_is_progress(self):
        """A prefill sends one output per request, after KV was released on success."""
        tm, clock = _attached(DisaggregationMode.PREFILL)
        tm.rid_to_state = {"a": _state()}
        hg.track_dispatched(tm, "a")
        clock.advance(5)
        hg.track_outputs(tm, _outputs(("a", None)))  # not a final output
        self.assertIsNone(hg.state_of(tm).tracker.snapshot().last_progress_age_s)
        hg.track_outputs(tm, _outputs(("a", LENGTH)))
        snap = hg.state_of(tm).tracker.snapshot()
        self.assertEqual((snap.requests_present, snap.last_progress_age_s), (0, 0.0))
        self.assertIsNone(snap.last_failure_age_s)

    def test_engine_abort_is_a_failure_and_ends_the_request(self):
        tm, clock = _attached()
        tm.rid_to_state = {"a": _state()}
        hg.track_dispatched(tm, "a")
        clock.advance(60)
        hg.track_outputs(tm, _outputs(("a", ABORT)))
        snap = hg.state_of(tm).tracker.snapshot()
        self.assertEqual(snap.requests_present, 0)
        self.assertEqual(snap.last_failure_age_s, 0.0)
        self.assertIsNone(snap.last_progress_age_s)
        self.assertEqual(
            snap.requests_age_s, 60.0, "the failure keeps requests outstanding"
        )

    def test_one_batch_output_can_carry_several_requests(self):
        tm, clock = _attached()
        tm.rid_to_state = {r: _state() for r in "abc"}
        for r in "abc":
            hg.track_dispatched(tm, r)
        hg.track_outputs(
            tm, _outputs(("a", None), ("b", LENGTH), ("c", ABORT), ("zz", ABORT))
        )
        snap = hg.state_of(tm).tracker.snapshot()
        self.assertEqual(snap.requests_present, 1, "a still in flight; zz unknown")
        self.assertEqual(snap.last_progress_age_s, 0.0)
        self.assertEqual(snap.last_failure_age_s, 0.0)

    def test_client_cancel_is_no_longer_outstanding_and_records_nothing(self):
        tm, clock = _attached()
        tm.rid_to_state = {"a": _state(), "b": _state()}
        hg.track_dispatched(tm, "a")
        hg.track_dispatched(tm, "b")
        hg.track_cancelled(tm, rid="a", abort_all=False)
        self.assertEqual(hg.state_of(tm).tracker.snapshot().requests_present, 1)
        # The scheduler's abort output for a cancelled rid arrives later: ignored.
        hg.track_outputs(tm, _outputs(("a", ABORT)))
        self.assertIsNone(hg.state_of(tm).tracker.snapshot().last_failure_age_s)
        hg.track_cancelled(tm, rid="", abort_all=True)
        snap = hg.state_of(tm).tracker.snapshot()
        self.assertEqual(snap.requests_present, 0)
        self.assertIsNone(snap.requests_age_s)

    def test_requests_whose_state_vanished_are_no_longer_outstanding(self):
        """Any path that deletes a ReqState (disconnect, timeout) drops the request."""
        tm, clock = _attached()
        tm.rid_to_state = {"a": _state(), "b": _state()}
        hg.track_dispatched(tm, "a")
        hg.track_dispatched(tm, "b")
        del tm.rid_to_state["a"]
        hg.track_outputs(tm, _outputs(("b", None)))  # any event re-syncs the set
        self.assertEqual(hg.state_of(tm).tracker.snapshot().requests_present, 1)

    def test_bookkeeping_is_a_no_op_on_a_manager_without_a_tracker(self):
        tm = _tm(DisaggregationMode.NULL)
        tm.rid_to_state = {"a": _state()}
        hg.track_dispatched(tm, "a")
        hg.track_outputs(tm, _outputs(("a", ABORT)))
        hg.track_cancelled(tm, rid="a", abort_all=False)
        self.assertIsNone(hg.state_of(tm))

    def test_hung_scheduler_is_judged_from_this_side(self):
        """Requests go out and nothing comes back: the verdict flips without the
        scheduler running any code."""
        tm, clock = _attached()
        tm.rid_to_state = {"a": _state(), "b": _state()}
        tracker = hg.state_of(tm).tracker
        tracker.record_progress()  # warm
        hg.track_dispatched(tm, "a")
        hg.track_dispatched(tm, "b")
        clock.advance(CFG.window_s)
        hg._observe(tm, hg.state_of(tm))
        self.assertEqual(tracker.health_state(), "unhealthy")


# ---------------------------------------------------------------------------
# Per-peer judgement on a decode
# ---------------------------------------------------------------------------

P1, P2, P3 = "10.0.0.1:8998", "10.0.0.2:8998", "10.0.0.3:8998"


def _peers(ratio=0.5, streak=3, peer_min=1):
    clock = _Clock()
    cfg = PDHealthConfig(
        window_s=100.0,
        cold_window_s=250.0,
        peer_failure_ratio=ratio,
        peer_failure_streak=streak,
        peer_min=peer_min,
    )
    return PDPeerHealth(config=cfg, clock=clock), clock


def _fail(peers, peer, n):
    for _ in range(n):
        peers.record_failure(peer)


class TestPeers(CustomTestCase):
    def test_no_peers_is_ready(self):
        peers, _ = _peers()
        self.assertEqual(peers.health_state(), "ready")
        self.assertEqual(peers.snapshot().active_peers, 0)

    def test_one_abort_is_noise_but_a_streak_marks_the_peer_failing(self):
        peers, _ = _peers(streak=3)
        _fail(peers, P1, 2)
        self.assertEqual(peers.snapshot().peers[0].state, "ok")
        _fail(peers, P1, 1)
        snap = peers.snapshot()
        self.assertEqual(
            (snap.peers[0].state, snap.peers[0].failure_streak), ("failing", 3)
        )

    def test_a_success_resets_the_streak(self):
        peers, _ = _peers(streak=3)
        _fail(peers, P1, 2)
        peers.record_progress(P1)
        _fail(peers, P1, 2)
        self.assertEqual(peers.snapshot().peers[0].state, "ok")

    def test_ratio_of_failing_peers_decides(self):
        peers, _ = _peers(ratio=0.5)
        peers.record_progress(P1)
        peers.record_progress(P2)
        peers.record_progress(P3)
        _fail(peers, P1, 3)
        self.assertEqual(peers.health_state(), "ready", "1 of 3 is under 50%")
        _fail(peers, P2, 3)
        self.assertEqual(peers.health_state(), "unhealthy", "2 of 3 reaches 50%")
        snap = peers.snapshot()
        self.assertEqual((snap.active_peers, snap.failing_peers), (3, 2))
        self.assertIn(
            "peers failing: 2 of 3 active (threshold 50%, 3 aborts in a row or 100s in flight with nothing back): 10.0.0.1:8998, 10.0.0.2:8998",
            snap.reason,
        )

    def test_threshold_is_inclusive_and_configurable(self):
        peers, _ = _peers(ratio=1.0)
        peers.record_progress(P1)
        _fail(peers, P2, 3)
        self.assertEqual(peers.health_state(), "ready", "1 of 2 is under 100%")
        _fail(peers, P1, 3)
        self.assertEqual(peers.health_state(), "unhealthy")

    def test_single_peer_pair_trips_on_its_own_streak(self):
        peers, _ = _peers(ratio=0.5, peer_min=1)
        _fail(peers, P1, 3)
        self.assertEqual(peers.health_state(), "unhealthy")

    def test_peer_min_withholds_judgement_on_too_few_peers(self):
        peers, _ = _peers(ratio=0.5, peer_min=2)
        _fail(peers, P1, 3)
        self.assertEqual(peers.health_state(), "ready")
        peers.record_progress(P2)
        self.assertEqual(peers.health_state(), "unhealthy", "1 of 2 reaches 50%")

    def test_idle_peers_age_out_of_the_active_set(self):
        peers, clock = _peers()
        _fail(peers, P1, 3)
        peers.record_progress(P2)
        clock.advance(100)
        self.assertEqual(peers.snapshot().active_peers, 0)
        self.assertEqual(peers.health_state(), "ready")
        peers.observe({P1: 1})  # in flight keeps a peer active past the window
        self.assertEqual(peers.snapshot().active_peers, 1)

    def test_a_failing_peers_last_abort_ages_out(self):
        peers, clock = _peers()
        _fail(peers, P1, 3)
        clock.advance(99)
        peers.observe({P1: 1})  # a fresh request keeps the peer active
        self.assertEqual(peers.snapshot().peers[0].state, "failing")
        clock.advance(1)
        self.assertEqual(peers.snapshot().peers[0].state, "ok", "abort aged out")
        self.assertEqual(peers.health_state(), "ready")

    def test_requests_stuck_in_flight_for_a_window_mark_the_peer_failing(self):
        """Live finding: requests to a broken prefill hung until the client gave up,
        so no abort was ever recorded and the peer's streak aged out while its
        requests were still stuck."""
        peers, clock = _peers()
        peers.record_progress(P2)
        peers.observe({P1: 4})
        clock.advance(99)
        peers.observe({P1: 4})
        self.assertEqual(peers.snapshot().peers[0].state, "ok")
        clock.advance(1)
        self.assertEqual(peers.snapshot().peers[0].state, "failing")
        self.assertEqual(peers.health_state(), "unhealthy", "1 of 2 reaches 50%")
        peers.record_progress(P1)  # something came back from it
        self.assertEqual(peers.snapshot().peers[0].state, "ok")
        peers.observe({})  # drained: the in-flight clock resets
        clock.advance(100)
        peers.observe({P1: 1})
        self.assertEqual(peers.snapshot().peers[0].state, "ok", "fresh window")

    def test_ratio_zero_turns_the_rule_off(self):
        peers, _ = _peers(ratio=0.0)
        _fail(peers, P1, 10)
        self.assertEqual(peers.health_state(), "ready")

    def test_snapshot_lists_active_peers_with_their_facts(self):
        peers, clock = _peers()
        peers.record_progress(P1)
        clock.advance(7)
        _fail(peers, P2, 1)
        peers.observe({P2: 2})
        snap = peers.snapshot()
        self.assertEqual([p.peer for p in snap.peers], [P1, P2])
        self.assertEqual(
            (snap.peers[0].last_progress_age_s, snap.peers[0].in_flight), (7.0, 0)
        )
        self.assertEqual(
            (
                snap.peers[1].last_failure_age_s,
                snap.peers[1].in_flight,
                snap.peers[1].failure_streak,
            ),
            (0.0, 2, 1),
        )
        self.assertEqual(json_roundtrip(snap)["failure_ratio"], 0.5)

    def test_invalid_config_is_rejected(self):
        for kw in (
            {"peer_failure_ratio": 1.5},
            {"peer_failure_streak": 0},
            {"peer_min": 0},
        ):
            with self.assertRaises(ValueError):
                PDHealthConfig(window_s=1.0, cold_window_s=1.0, **kw).validate()

    def test_build_is_decode_only_and_honours_the_knobs(self):
        self.assertIsNone(build_pd_peer_tracker("prefill"))
        self.assertIsNotNone(build_pd_peer_tracker("decode"))
        with tpu_envs.SGLANG_TPU_PD_HEALTH_PEER_FAILURE_RATIO.override(0.0):
            self.assertIsNone(build_pd_peer_tracker("decode"))
        with tpu_envs.SGLANG_TPU_PD_HEALTH_READINESS_ENABLED.override(False):
            self.assertIsNone(build_pd_peer_tracker("decode"))
        with (
            tpu_envs.SGLANG_TPU_PD_HEALTH_PEER_FAILURE_RATIO.override(0.75),
            tpu_envs.SGLANG_TPU_PD_HEALTH_PEER_FAILURE_STREAK.override(5),
            tpu_envs.SGLANG_TPU_PD_HEALTH_PEER_MIN.override(2),
        ):
            cfg = build_pd_peer_tracker("decode").config
        self.assertEqual(
            (cfg.peer_failure_ratio, cfg.peer_failure_streak, cfg.peer_min),
            (0.75, 5, 2),
        )


def json_roundtrip(snap):
    return msgspec.json.decode(snap.to_json())


def _attached_with_peers(role=DisaggregationMode.DECODE):
    tm = _tm(role)
    clock = _Clock()
    cfg = PDHealthConfig(window_s=100.0, cold_window_s=250.0, peer_failure_streak=2)
    hg._STATE[tm] = hg._HttpHealth(
        PDProgressHealth(config=CFG, clock=clock, role=role.value),
        PDPeerHealth(config=cfg, clock=clock),
    )
    return tm, clock


class TestPeerBookkeeping(CustomTestCase):
    def test_dispatch_attributes_the_request_to_the_router_stamped_prefill(self):
        tm, clock = _attached_with_peers()
        tm.rid_to_state = {
            "a": _state(P1),
            "b": _state(P1),
            "c": _state(P2),
            "p": _state(None),
        }
        for rid in "abcp":
            hg.track_dispatched(tm, rid)
        peers = hg.state_of(tm).peers
        snap = peers.snapshot()
        self.assertEqual({p.peer: p.in_flight for p in snap.peers}, {P1: 2, P2: 1})
        self.assertEqual(hg.state_of(tm).rid_peer, {"a": P1, "b": P1, "c": P2})

    def test_outcomes_land_on_the_right_peer(self):
        tm, clock = _attached_with_peers()
        tm.rid_to_state = {r: _state(P1 if r in "ab" else P2) for r in "abcd"}
        for rid in "abcd":
            hg.track_dispatched(tm, rid)
        hg.track_outputs(
            tm, _outputs(("a", ABORT), ("b", ABORT), ("c", None), ("d", LENGTH))
        )
        snap = hg.state_of(tm).peers.snapshot()
        by = {p.peer: p for p in snap.peers}
        self.assertEqual(
            (by[P1].state, by[P1].failure_streak, by[P1].in_flight), ("failing", 2, 0)
        )
        self.assertEqual(
            (by[P2].state, by[P2].failure_streak, by[P2].in_flight), ("ok", 0, 1)
        )
        self.assertEqual(hg.state_of(tm).rid_peer, {"c": P2})

    def test_cancel_and_vanished_state_drop_the_peer_mapping(self):
        tm, clock = _attached_with_peers()
        tm.rid_to_state = {"a": _state(P1), "b": _state(P2)}
        hg.track_dispatched(tm, "a")
        hg.track_dispatched(tm, "b")
        hg.track_cancelled(tm, rid="a", abort_all=False)
        del tm.rid_to_state["b"]
        hg.track_outputs(tm, _outputs(("b", None)))  # any tracked event re-syncs
        self.assertEqual(hg.state_of(tm).rid_peer, {})
        self.assertEqual(
            sum(p.in_flight for p in hg.state_of(tm).peers.snapshot().peers), 0
        )

    def test_prefill_has_no_peer_tracker(self):
        tm = _tm(DisaggregationMode.PREFILL)
        with patch.object(hg, "install_health_generate_route", lambda: True):
            hg.attach_tracker(tm)
        self.assertIsNone(hg.state_of(tm).peers)
        tm.rid_to_state = {"a": _state(P1)}
        hg.track_dispatched(tm, "a")  # must not touch peers
        self.assertEqual(hg.state_of(tm).rid_peer, {})

    def test_decode_gets_a_peer_tracker_by_default(self):
        tm = _tm(DisaggregationMode.DECODE)
        with patch.object(hg, "install_health_generate_route", lambda: True):
            hg.attach_tracker(tm)
        self.assertIsNotNone(hg.state_of(tm).peers)

    def test_failing_peers_give_503_while_the_engine_rule_is_still_ready(self):
        """Two prefills, one broken: the engine still produces, so the stall rule
        is ready; the peer rule says half the peers are failing."""
        tm, clock = _attached_with_peers()
        tm.rid_to_state = {r: _state(P1 if r in "ab" else P2) for r in "abcd"}
        for rid in "abcd":
            hg.track_dispatched(tm, rid)
        hg.track_outputs(
            tm, _outputs(("a", ABORT), ("b", ABORT), ("c", LENGTH), ("d", LENGTH))
        )
        state = hg.state_of(tm)
        self.assertEqual(state.tracker.health_state(), "ready")
        response = hg.health_response(state.tracker, state.peers)
        self.assertEqual(response.status_code, 503)
        body = msgspec.json.decode(response.body)
        self.assertIn("peers failing: 1 of 2 active", body["reason"])
        self.assertEqual(body["pd_health"]["state"], "ready")
        self.assertEqual(body["pd_peers"]["failing_peers"], 1)
        self.assertEqual(
            [p["peer"] for p in body["pd_peers"]["peers"] if p["state"] == "failing"],
            [P1],
        )

    def test_engine_rule_reason_comes_first_when_both_trip(self):
        tm, clock = _attached_with_peers()
        tm.rid_to_state = {"a": _state(P1)}
        hg.track_dispatched(tm, "a")
        clock.advance(CFG.cold_window_s)
        hg._observe(tm, hg.state_of(tm))
        state = hg.state_of(tm)
        body = msgspec.json.decode(hg.health_response(state.tracker, state.peers).body)
        self.assertIn("stalled: 1 requests held", body["reason"])
        self.assertEqual(
            body["pd_peers"]["state"], "unhealthy", "the peer is stalled too"
        )
        self.assertEqual(body["pd_peers"]["peers"][0]["state"], "failing")

    def test_response_without_a_peer_tracker_carries_null_peers(self):
        tm, clock = _attached(DisaggregationMode.PREFILL)
        tm.rid_to_state = {"a": _state()}
        hg.track_dispatched(tm, "a")
        clock.advance(CFG.cold_window_s)
        hg._observe(tm, hg.state_of(tm))
        body = msgspec.json.decode(
            hg.health_response(hg.state_of(tm).tracker, None).body
        )
        self.assertIsNone(body["pd_peers"])


# ---------------------------------------------------------------------------
# The /health_generate route
# ---------------------------------------------------------------------------


class TestRoute(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._routes = list(http_server.app.router.routes)
        self._orig = dict(hg._ORIGINAL_HANDLER)
        self._prior_state = http_server.get_global_state()

    def tearDown(self):
        http_server.app.router.routes[:] = self._routes
        hg._ORIGINAL_HANDLER.clear()
        hg._ORIGINAL_HANDLER.update(self._orig)
        http_server._global_state = self._prior_state
        super().tearDown()

    def _route(self, path):
        from fastapi.routing import APIRoute

        return [
            r
            for r in http_server.app.router.routes
            if isinstance(r, APIRoute) and r.path == path
        ]

    def test_install_swaps_only_health_generate_and_is_idempotent(self):
        self.assertTrue(hg.install_health_generate_route())
        self.assertFalse(
            hg.install_health_generate_route(), "second install is a no-op"
        )
        (route,) = self._route(hg.HEALTH_GENERATE_PATH)
        self.assertIs(route.endpoint, hg.health_generate_with_tracker)
        self.assertEqual(sorted(route.methods), ["GET"])
        # Unannotated, FastAPI would read it as a required query field (422).
        self.assertIs(typing.get_type_hints(route.endpoint)["request"], Request)
        (h,) = self._route("/health")
        self.assertIs(
            h.endpoint, http_server.health_generate, "/health keeps the core handler"
        )
        self.assertIs(hg._ORIGINAL_HANDLER["handler"], http_server.health_generate)

    def _call(self, tm):
        http_server.set_global_state(SimpleNamespace(tokenizer_manager=tm))
        request = SimpleNamespace(url=SimpleNamespace(path=hg.HEALTH_GENERATE_PATH))
        return asyncio.run(hg.health_generate_with_tracker(request))

    def _stalled(self, role=DisaggregationMode.DECODE):
        tm, clock = _attached(role)
        tm.rid_to_state = {r: _state() for r in "abc"}
        for r in "abc":
            hg.track_dispatched(tm, r)
        clock.advance(CFG.cold_window_s)
        hg._observe(tm, hg.state_of(tm))
        return tm

    def test_stalled_engine_returns_503_with_the_facts(self):
        response = self._call(self._stalled())
        self.assertEqual(response.status_code, 503)
        body = msgspec.json.decode(response.body)
        self.assertEqual((body["role"], body["state"]), ("decode", "unhealthy"))
        self.assertIn("stalled: 3 requests held, outstanding for 250s", body["reason"])
        self.assertEqual(body["pd_health"]["requests_present"], 3)

    def test_prefill_is_gated_too(self):
        body = msgspec.json.decode(
            self._call(self._stalled(DisaggregationMode.PREFILL)).body
        )
        self.assertEqual(body["role"], "prefill")

    def test_ready_pd_engine_returns_200_without_a_probe(self):
        tm, _ = _attached()
        hg._ORIGINAL_HANDLER["handler"] = (
            None  # would raise if the probe path were taken
        )
        self.assertEqual(self._call(tm).status_code, 200)

    def test_shutdown_and_starting_gates_still_come_first(self):
        tm, _ = _attached()
        tm.gracefully_exit = True
        self.assertEqual(self._call(tm).status_code, 503)
        tm.gracefully_exit = False
        tm.server_status = ServerStatus.Starting
        self.assertEqual(self._call(tm).status_code, 503)

    def test_manager_without_a_tracker_delegates_to_the_core_handler(self):
        calls = []

        async def core_handler(request):
            calls.append(request.url.path)
            return SimpleNamespace(status_code=200)

        hg._ORIGINAL_HANDLER["handler"] = core_handler
        tm = _tm(DisaggregationMode.NULL)
        self.assertEqual(self._call(tm).status_code, 200)
        self.assertEqual(calls, [hg.HEALTH_GENERATE_PATH])

    def test_body_forwards_the_plugin_record_untouched(self):
        tm = self._stalled()
        body = msgspec.json.decode(self._call(tm).body)
        expected = msgspec.json.decode(hg.state_of(tm).tracker.health_json())
        self.assertEqual(body["pd_health"], expected)


if __name__ == "__main__":
    unittest.main()
