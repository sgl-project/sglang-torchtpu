"""Decode readiness from real transfer outcomes (sglang-torchtpu #1276, fix 1).

Black-box contract that must not come back: a decode whose every KV transfer
fails keeps answering 200 on /health_generate.

Usage:
    python3 -m pytest test/registered/unit/disaggregation/test_pd_transfer_health.py -v
"""

import asyncio
import unittest
from types import SimpleNamespace

import msgspec

from sglang.srt.disaggregation.health import (
    PDHealthConfig,
    PDHealthSnapshot,
    PDHealthState,
    PDTransferHealth,
)
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints import http_server
from sglang.srt.environ import envs
from sglang.srt.managers.load_snapshot import (
    DisaggregationMetrics,
    LoadSnapshot,
    snapshot_decoder,
    snapshot_encoder,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import ServerStatus, TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

# N=3 failures spanning T=100 s open a peer; more than half the peers open
# trips the engine; probation output deadline P=10 s; backoff cap 400 s.
CFG = PDHealthConfig(
    failure_threshold=3,
    failure_window_s=100.0,
    failure_ratio=0.5,
    probation_output_timeout_s=10.0,
    probation_backoff_cap_s=400.0,
)


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _tracker(config=CFG):
    clock = _Clock()
    return PDTransferHealth(config=config, clock=clock), clock


def _open_peer(tracker, clock, peer, n=3, step=50.0):
    """n failures on one peer, `step` seconds apart: spans (n-1)*step."""
    for _ in range(n):
        tracker.record_failure(peer=peer)
        clock.advance(step)


def _open_peers(tracker, clock, peers, n=3, step=50.0):
    """Interleaved failures so every peer reaches n with the same span."""
    for _ in range(n):
        for peer in peers:
            tracker.record_failure(peer=peer)
        clock.advance(step)


def _peer(snapshot, name):
    return next(p for p in snapshot.peers if p.peer == name)


class TestPeerCircuit(CustomTestCase):
    """Per peer: open iff N consecutive failures AND span >= T, no success between."""

    def test_opens_after_n_failures_spanning_at_least_t(self):
        tracker, clock = _tracker()
        _open_peer(tracker, clock, "p1", n=2, step=50.0)  # 2 failures, span 50
        self.assertFalse(_peer(tracker.snapshot(), "p1").open)
        tracker.record_failure(peer="p1")  # 3rd failure, span 100 == T
        self.assertTrue(_peer(tracker.snapshot(), "p1").open)

    def test_burst_shorter_than_t_never_opens_however_large(self):
        tracker, clock = _tracker()
        _open_peer(tracker, clock, "p1", n=60, step=1.0)  # span 59 < 100
        snap = tracker.snapshot()
        self.assertFalse(_peer(snap, "p1").open)
        self.assertEqual(_peer(snap, "p1").consecutive_failures, 60)
        self.assertEqual(snap.state, "ready")

    def test_a_success_inside_the_run_resets_count_and_span(self):
        tracker, clock = _tracker()
        _open_peer(tracker, clock, "p1", n=2, step=60.0)  # span 60
        tracker.record_success(peer="p1")
        clock.advance(60.0)
        tracker.record_failure(peer="p1")  # new run: 1 failure, span 0
        snap = _peer(tracker.snapshot(), "p1")
        self.assertFalse(snap.open)
        self.assertEqual(snap.consecutive_failures, 1)
        self.assertEqual(snap.failures_span_s, 0.0)

    def test_one_success_closes_an_open_peer(self):
        tracker, clock = _tracker()
        _open_peer(tracker, clock, "p1")
        self.assertTrue(_peer(tracker.snapshot(), "p1").open)
        tracker.record_success(peer="p1")
        self.assertFalse(_peer(tracker.snapshot(), "p1").open)

    def test_threshold_one_with_positive_window_never_opens_and_warns(self):
        cfg = PDHealthConfig(failure_threshold=1, failure_window_s=10.0)
        with self.assertLogs("sglang.srt.disaggregation.health", level="WARNING"):
            tracker = PDTransferHealth(config=cfg, clock=_Clock())
        tracker.record_failure(peer="p1")
        self.assertFalse(_peer(tracker.snapshot(), "p1").open)

    def test_threshold_one_with_zero_window_opens_on_first_failure(self):
        cfg = PDHealthConfig(failure_threshold=1, failure_window_s=0.0)
        clock = _Clock()
        tracker = PDTransferHealth(config=cfg, clock=clock)
        tracker.record_failure(peer="p1")
        self.assertEqual(tracker.state, PDHealthState.UNHEALTHY)
        snap = tracker.snapshot()
        self.assertTrue(_peer(snap, "p1").open)
        # A zero window must not mean a zero quiet wait: the decode stays out at
        # least for the probation output timeout, so the probe sees the 503.
        self.assertEqual(snap.state, "unhealthy")
        self.assertEqual(snap.probation_wait_s, cfg.probation_output_timeout_s)
        clock.advance(cfg.probation_output_timeout_s)
        self.assertEqual(tracker.tick(), PDHealthState.PROBATION)

    def test_invalid_config_is_rejected(self):
        for bad in (
            dict(failure_threshold=0),
            dict(failure_ratio=1.0),
            dict(failure_ratio=-0.1),
            dict(probation_output_timeout_s=0.0),
            dict(probation_backoff_cap_s=10.0, failure_window_s=60.0),
            dict(expected_peers=-1),
        ):
            with self.assertRaises(ValueError, msg=bad):
                PDTransferHealth(config=PDHealthConfig(**bad), clock=_Clock())


class TestPeerRatio(CustomTestCase):
    """Engine unhealthy iff open_peers / known_peers > ratio (strictly)."""

    def test_one_open_peer_of_two_stays_ready_and_degraded(self):
        tracker, clock = _tracker()
        tracker.record_success(peer="p2")
        _open_peer(tracker, clock, "p1")
        snap = tracker.snapshot()
        self.assertEqual(snap.state, "ready")
        self.assertTrue(snap.ready)
        self.assertEqual((snap.open_peers, snap.known_peers), (1, 2))
        self.assertIn("degraded", snap.reason)
        self.assertIn("p1", snap.reason)
        self.assertEqual(snap.open_peer_names, ["p1"])

    def test_two_open_peers_of_two_trip(self):
        tracker, clock = _tracker()
        _open_peers(tracker, clock, ["p1", "p2"])
        snap = tracker.snapshot()
        self.assertEqual(snap.state, "unhealthy")
        self.assertFalse(snap.ready)
        self.assertIn("peer_failure_ratio: 2/2", snap.reason)

    def test_ratio_boundary_is_strict(self):
        """2 of 4 is exactly 0.5 and must not trip; 2 of 3 must."""
        tracker, clock = _tracker()
        tracker.record_success(peer="p3")
        tracker.record_success(peer="p4")
        _open_peers(tracker, clock, ["p1", "p2"])
        self.assertEqual(tracker.state, PDHealthState.READY)

        tracker, clock = _tracker()
        tracker.record_success(peer="p3")
        _open_peers(tracker, clock, ["p1", "p2"])
        self.assertEqual(tracker.state, PDHealthState.UNHEALTHY)

    def test_single_known_peer_trips_alone(self):
        """1P1D: the only peer open is 1/1."""
        tracker, clock = _tracker()
        _open_peer(tracker, clock, "p1")
        self.assertEqual(tracker.state, PDHealthState.UNHEALTHY)

    def test_expected_peers_floor_prevents_the_startup_false_positive(self):
        cfg = msgspec.structs.replace(CFG, expected_peers=2)
        tracker, clock = _tracker(cfg)
        _open_peer(tracker, clock, "p1")  # only p1 seen so far
        snap = tracker.snapshot()
        self.assertEqual(snap.state, "ready")
        self.assertEqual(snap.known_peers, 2)
        _open_peer(tracker, clock, "p2")
        self.assertEqual(tracker.state, PDHealthState.UNHEALTHY)

    def test_expected_peers_floor_never_shrinks_an_observed_fleet(self):
        cfg = msgspec.structs.replace(CFG, expected_peers=1)
        tracker, clock = _tracker(cfg)
        tracker.record_success(peer="p2")
        _open_peer(tracker, clock, "p1")
        self.assertEqual(tracker.snapshot().known_peers, 2)
        self.assertEqual(tracker.state, PDHealthState.READY)

    def test_no_peers_is_ready(self):
        tracker, _ = _tracker()
        snap = tracker.snapshot()
        self.assertEqual(
            (snap.state, snap.known_peers, snap.reason), ("ready", 0, "ready")
        )

    def test_failures_on_one_peer_do_not_count_against_another(self):
        """Fifty failures on p1 must not open p2, however the run interleaves."""
        tracker, clock = _tracker()
        for _ in range(50):
            tracker.record_failure(peer="p1")
            tracker.record_success(peer="p2")
            clock.advance(5.0)
        snap = tracker.snapshot()
        self.assertTrue(_peer(snap, "p1").open)
        self.assertFalse(_peer(snap, "p2").open)
        self.assertEqual(snap.state, "ready")


class TestProbation(CustomTestCase):
    def _unhealthy(self):
        tracker, clock = _tracker()
        _open_peers(tracker, clock, ["p1", "p2"])
        self.assertEqual(tracker.state, PDHealthState.UNHEALTHY)
        return tracker, clock

    def test_probation_after_t_with_no_new_failure(self):
        tracker, clock = self._unhealthy()
        clock.advance(CFG.failure_window_s - 50.0 - 1.0)  # last failure was 50 s ago
        self.assertEqual(tracker.tick(), PDHealthState.UNHEALTHY)
        clock.advance(1.0)
        self.assertEqual(tracker.tick(), PDHealthState.PROBATION)
        snap = tracker.snapshot()
        self.assertTrue(snap.ready, "probation answers 200 so traffic resumes")
        self.assertIn("probation", snap.reason)

    def test_new_failure_during_the_quiet_wait_restarts_it(self):
        tracker, clock = self._unhealthy()
        clock.advance(40.0)
        tracker.record_failure(peer="p1")
        clock.advance(CFG.failure_window_s - 1.0)
        self.assertEqual(tracker.tick(), PDHealthState.UNHEALTHY)
        clock.advance(1.0)
        self.assertEqual(tracker.tick(), PDHealthState.PROBATION)

    def test_admitted_request_without_output_within_p_fails_again(self):
        tracker, clock = self._unhealthy()
        clock.advance(CFG.failure_window_s)
        tracker.tick()
        tracker.record_admitted()
        clock.advance(CFG.probation_output_timeout_s)
        self.assertEqual(tracker.tick(), PDHealthState.PROBATION, "deadline inclusive")
        clock.advance(0.1)
        self.assertEqual(tracker.tick(), PDHealthState.UNHEALTHY)
        self.assertEqual(tracker.snapshot().probation_wait_s, 200.0, "doubled")

    def test_missed_deadline_starts_the_quiet_wait_and_doubles_once(self):
        """Live 2P/2D run: a missed deadline was followed by the request's own
        failure 60 s later, and the wait went 60 -> 240 because the stale
        last-failure time re-admitted the decode in between."""
        tracker, clock = self._unhealthy()
        clock.advance(CFG.failure_window_s)
        tracker.tick()
        tracker.record_admitted()
        clock.advance(CFG.probation_output_timeout_s + 1)
        self.assertEqual(tracker.tick(), PDHealthState.UNHEALTHY)
        self.assertEqual(tracker.snapshot().probation_wait_s, 200.0)
        clock.advance(1.0)
        self.assertEqual(tracker.tick(), PDHealthState.UNHEALTHY, "not re-admitted")
        tracker.record_failure(peer="p1")  # the admitted request finally fails
        self.assertEqual(tracker.snapshot().probation_wait_s, 200.0, "doubled once")
        clock.advance(200.0)
        self.assertEqual(tracker.tick(), PDHealthState.PROBATION)

    def test_backoff_doubles_up_to_the_cap(self):
        tracker, clock = self._unhealthy()
        waits = []
        for _ in range(4):
            clock.advance(tracker.snapshot().probation_wait_s)
            self.assertEqual(tracker.tick(), PDHealthState.PROBATION)
            tracker.record_failure(peer="p1")  # probation request fails
            waits.append(tracker.snapshot().probation_wait_s)
        self.assertEqual(waits, [200.0, 400.0, 400.0, 400.0])

    def test_probation_success_that_clears_the_ratio_returns_ready(self):
        tracker, clock = self._unhealthy()
        clock.advance(CFG.failure_window_s)
        tracker.tick()
        tracker.record_admitted()
        clock.advance(5.0)
        tracker.record_success(peer="p1")  # 1/2 open: ratio no longer trips
        self.assertEqual(tracker.state, PDHealthState.READY)
        snap = tracker.snapshot()
        self.assertEqual(snap.probation_wait_s, CFG.failure_window_s, "backoff reset")
        self.assertIsNone(snap.probation_deadline_in_s)
        self.assertIn("degraded", snap.reason, "p2 is still open")

    def test_probation_success_that_leaves_the_ratio_tripped_stays_on_probation(self):
        """10 peers, 8 open: one good transfer re-arms the deadline instead of closing."""
        tracker, clock = _tracker()
        tracker.record_success(peer="p9")
        tracker.record_success(peer="p10")
        _open_peers(tracker, clock, [f"p{i}" for i in range(1, 9)])
        self.assertEqual(tracker.state, PDHealthState.UNHEALTHY)
        clock.advance(CFG.failure_window_s)
        tracker.tick()
        tracker.record_admitted()
        tracker.record_success(peer="p1")  # 7/10 still > 0.5
        self.assertEqual(tracker.state, PDHealthState.PROBATION)
        self.assertIsNone(tracker.snapshot().probation_deadline_in_s)
        clock.advance(CFG.probation_output_timeout_s + 1)
        self.assertEqual(tracker.tick(), PDHealthState.PROBATION, "no deadline armed")
        tracker.record_admitted()
        clock.advance(CFG.probation_output_timeout_s + 1)
        self.assertEqual(tracker.tick(), PDHealthState.UNHEALTHY)

    def test_idle_probation_stays_ready_indefinitely(self):
        tracker, clock = self._unhealthy()
        clock.advance(CFG.failure_window_s)
        tracker.tick()
        clock.advance(24 * 3600.0)
        self.assertEqual(tracker.tick(), PDHealthState.PROBATION)
        self.assertTrue(tracker.snapshot().ready)

    def test_admitted_is_ignored_outside_probation(self):
        tracker, clock = self._unhealthy()
        tracker.record_admitted()
        self.assertIsNone(tracker.snapshot().probation_deadline_in_s)
        ready, _ = _tracker()
        ready.record_admitted()
        self.assertIsNone(ready.snapshot().probation_deadline_in_s)

    def test_traffic_that_never_stopped_recovers_without_probation(self):
        """A router that ignores 503 keeps sending; successes close peers directly."""
        tracker, clock = self._unhealthy()
        clock.advance(5.0)
        tracker.record_success(peer="p1")
        self.assertEqual(tracker.state, PDHealthState.READY)

    def test_version_counts_state_changes_only(self):
        tracker, clock = self._unhealthy()
        v = tracker.version
        self.assertEqual(v, 1)
        tracker.record_failure(peer="p1")  # still unhealthy
        self.assertEqual(tracker.version, v)
        clock.advance(CFG.failure_window_s)
        tracker.tick()
        self.assertEqual(tracker.version, v + 1)


class TestSnapshotWire(CustomTestCase):
    def test_health_round_trips_through_the_load_snapshot(self):
        tracker, clock = _tracker()
        _open_peers(tracker, clock, ["p1", "p2"])
        health = tracker.snapshot()
        load = LoadSnapshot(
            dp_rank=0,
            disaggregation=DisaggregationMetrics(
                mode="decode", health_state=health.state, health_json=health.to_json()
            ),
        )
        back = snapshot_decoder.decode(snapshot_encoder.encode(load))
        self.assertEqual(back.disaggregation.health_state, "unhealthy")
        restored = PDHealthSnapshot.from_json(back.disaggregation.health_json)
        self.assertEqual(restored, health)
        self.assertEqual(restored.open_peer_names, ["p1", "p2"])

    def test_default_disaggregation_metrics_carry_no_health(self):
        self.assertEqual(DisaggregationMetrics(mode="prefill").health_json, "")

    def test_from_env_reads_every_knob(self):
        with (
            envs.SGLANG_PD_HEALTH_FAILURE_THRESHOLD.override(7),
            envs.SGLANG_PD_HEALTH_FAILURE_WINDOW_S.override(12.5),
            envs.SGLANG_PD_HEALTH_FAILURE_RATIO.override(0.75),
            envs.SGLANG_PD_HEALTH_PROBATION_OUTPUT_TIMEOUT_S.override(3.0),
            envs.SGLANG_PD_HEALTH_PROBATION_BACKOFF_CAP_S.override(99.0),
            envs.SGLANG_PD_HEALTH_EXPECTED_PEERS.override(4),
        ):
            cfg = PDHealthConfig.from_env()
        self.assertEqual(
            cfg,
            PDHealthConfig(
                failure_threshold=7,
                failure_window_s=12.5,
                failure_ratio=0.75,
                probation_output_timeout_s=3.0,
                probation_backoff_cap_s=99.0,
                expected_peers=4,
            ),
        )


class TestQueueWiring(CustomTestCase):
    def test_outcomes_are_recorded_by_peer_and_probes_are_skipped(self):
        from sglang.srt.disaggregation.decode import (
            _record_pd_admission,
            _record_pd_transfer_outcome,
        )

        calls = []
        fake = SimpleNamespace(
            record_success=lambda peer: calls.append(("ok", peer)),
            record_failure=lambda peer: calls.append(("fail", peer)),
            record_admitted=lambda: calls.append(("admitted",)),
        )
        req = SimpleNamespace(bootstrap_host="10.0.0.1", bootstrap_port=8998)
        probe = SimpleNamespace(bootstrap_host=FAKE_BOOTSTRAP_HOST, bootstrap_port=8998)
        _record_pd_transfer_outcome(fake, req, ok=False)
        _record_pd_transfer_outcome(fake, req, ok=True)
        _record_pd_admission(fake, req)
        _record_pd_transfer_outcome(fake, probe, ok=False)
        _record_pd_admission(fake, probe)
        # No tracker (prefill, or readiness disabled): a no-op.
        _record_pd_transfer_outcome(None, req, ok=False)
        _record_pd_admission(None, req)
        self.assertEqual(
            calls,
            [("fail", "10.0.0.1:8998"), ("ok", "10.0.0.1:8998"), ("admitted",)],
        )

    def test_queues_default_to_no_tracker(self):
        """Fixtures that build a queue without a scheduler double must keep working."""
        from sglang.srt.disaggregation.decode import DecodeTransferQueue

        self.assertIsNone(DecodeTransferQueue.__init__.__defaults__[-1]())

    def test_maybe_init_builds_the_tracker_for_decode_only(self):
        from sglang.srt.disaggregation.utils import DisaggregationMode

        self.assertIsNone(Scheduler.pd_health, "class default keeps doubles safe")
        with envs.SGLANG_PD_HEALTH_FAILURE_THRESHOLD.override(7):
            for mode, expect in (
                (DisaggregationMode.DECODE, True),
                (DisaggregationMode.PREFILL, False),
                (DisaggregationMode.NULL, False),
            ):
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.disaggregation_mode = mode
                scheduler.maybe_init_pd_health()
                self.assertEqual(scheduler.pd_health is not None, expect, mode)
                if expect:
                    self.assertEqual(scheduler.pd_health.config.failure_threshold, 7)
        with envs.SGLANG_ENABLE_PD_HEALTH_READINESS.override(False):
            scheduler = Scheduler.__new__(Scheduler)
            scheduler.disaggregation_mode = DisaggregationMode.DECODE
            scheduler.maybe_init_pd_health()
            self.assertIsNone(scheduler.pd_health)

    def test_idle_transition_forces_a_publish_exactly_once(self):
        """Unhealthy -> probation happens on the clock; the probe must see it."""
        tracker, clock = _tracker()
        _open_peers(tracker, clock, ["p1", "p2"])
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.pd_health = tracker
        published = []
        scheduler.publish_load_snapshot = lambda force=False: published.append(force)
        scheduler.maybe_publish_pd_health()  # first sight of the unhealthy state
        scheduler.maybe_publish_pd_health()  # nothing changed
        self.assertEqual(published, [True])
        clock.advance(CFG.failure_window_s)
        scheduler.maybe_publish_pd_health()  # ticks into probation
        self.assertEqual(published, [True, True])
        self.assertEqual(tracker.state, PDHealthState.PROBATION)
        untracked = Scheduler.__new__(Scheduler)
        untracked.publish_load_snapshot = lambda force=False: published.append("no")
        untracked.maybe_publish_pd_health()
        self.assertEqual(published, [True, True])


class _FakeReader:
    def __init__(self, loads):
        self.loads = loads

    def read_all(self):
        return self.loads


def _load(health=None, mode="decode"):
    disagg = DisaggregationMetrics(mode=mode)
    if health is not None:
        disagg.health_state = health.state
        disagg.health_json = health.to_json()
    return LoadSnapshot(dp_rank=0, disaggregation=disagg)


class TestHealthGenerateGate(CustomTestCase):
    def _tm(self, loads):
        tm = TokenizerManager.__new__(TokenizerManager)
        tm.gracefully_exit = False
        tm.server_status = ServerStatus.Up
        tm.is_generation = True
        tm.rid_to_state = {}
        tm.last_receive_tstamp = 0.0
        tm.load_snapshot_reader = _FakeReader(loads)
        return tm

    def _call(self, tm, mode="decode"):
        prior_state = http_server.get_global_state()
        http_server.set_global_state(SimpleNamespace(tokenizer_manager=tm))
        prior_get_disagg = http_server.get_disagg
        http_server.get_disagg = lambda: SimpleNamespace(disaggregation_mode=mode)
        try:
            request = SimpleNamespace(url=SimpleNamespace(path="/health_generate"))
            return asyncio.run(http_server.health_generate(request))
        finally:
            http_server.get_disagg = prior_get_disagg
            http_server._global_state = prior_state

    def test_unhealthy_decode_returns_503_with_the_peers_before_probing(self):
        tracker, clock = _tracker()
        _open_peers(tracker, clock, ["p1", "p2"])
        tm = self._tm([_load(tracker.snapshot())])
        tm.generate_request = None  # the probe must not even be built
        response = self._call(tm)
        self.assertEqual(response.status_code, 503)
        body = msgspec.json.decode(response.body)
        self.assertEqual(body["state"], "unhealthy")
        self.assertIn("peer_failure_ratio: 2/2", body["reason"])
        self.assertEqual(body["known_peers"], 2)
        self.assertEqual(body["open_peers"], ["p1", "p2"])
        self.assertEqual(
            [p["peer"] for p in body["pd_health"][0]["peers"] if p["open"]],
            ["p1", "p2"],
        )

    def test_one_unready_rank_among_many_is_enough(self):
        tracker, clock = _tracker()
        _open_peers(tracker, clock, ["p1", "p2"])
        tm = self._tm([_load(PDHealthSnapshot()), _load(tracker.snapshot())])
        prior = http_server.get_disagg
        http_server.get_disagg = lambda: SimpleNamespace(disaggregation_mode="decode")
        try:
            self.assertEqual(len(http_server.pd_health_unready_snapshots(tm)), 1)
        finally:
            http_server.get_disagg = prior

    def test_ready_probation_and_untracked_pass_the_gate(self):
        tracker, clock = _tracker()
        _open_peers(tracker, clock, ["p1", "p2"])
        clock.advance(CFG.failure_window_s)
        probation = tracker.snapshot()
        self.assertEqual(probation.state, "probation")
        degraded, dclock = _tracker()
        degraded.record_success(peer="p2")
        _open_peer(degraded, dclock, "p1")
        prior = http_server.get_disagg
        http_server.get_disagg = lambda: SimpleNamespace(disaggregation_mode="decode")
        try:
            for loads in (
                [_load(PDHealthSnapshot())],
                [_load(probation)],
                [_load(degraded.snapshot())],
                [_load(None)],
                [],
            ):
                self.assertEqual(
                    http_server.pd_health_unready_snapshots(self._tm(loads)), []
                )
        finally:
            http_server.get_disagg = prior

    def test_gate_is_inactive_outside_decode_mode(self):
        tracker, clock = _tracker()
        _open_peers(tracker, clock, ["p1", "p2"])
        tm = self._tm([_load(tracker.snapshot(), mode="prefill")])
        prior = http_server.get_disagg
        http_server.get_disagg = lambda: SimpleNamespace(disaggregation_mode="prefill")
        try:
            self.assertEqual(http_server.pd_health_unready_snapshots(tm), [])
        finally:
            http_server.get_disagg = prior

    def test_no_reader_passes_the_gate(self):
        tm = self._tm([])
        tm.load_snapshot_reader = None
        prior = http_server.get_disagg
        http_server.get_disagg = lambda: SimpleNamespace(disaggregation_mode="decode")
        try:
            self.assertEqual(http_server.pd_health_unready_snapshots(tm), [])
        finally:
            http_server.get_disagg = prior


if __name__ == "__main__":
    unittest.main()
