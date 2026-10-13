"""/health liveness from per-process loop heartbeats.

Black-box contract: a scheduler rank or the detokenizer that stops iterating
turns /health into 503 within the configured max age, without any request
being sent into the stuck process; a healthy idle engine never trips it.

Usage:
    python3 -m pytest test/registered/unit/managers/test_health_heartbeat.py -v
"""

import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace

import msgspec

from sglang.srt.entrypoints import http_server
from sglang.srt.environ import envs
from sglang.srt.managers.heartbeat import (
    DETOKENIZER_SLOT,
    ROLE_DETOKENIZER,
    ROLE_SCHEDULER,
    SLOT_CAPACITY,
    HeartbeatReader,
    HeartbeatSlot,
    HeartbeatWriter,
    create_heartbeat_reader,
    create_heartbeat_writer,
    heartbeat_shm_path,
    scheduler_slot,
)
from sglang.srt.managers.tokenizer_manager import ServerStatus, TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Clock:
    def __init__(self):
        self.t = 5000.0

    def __call__(self):
        return self.t


class TestHeartbeatBoard(CustomTestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "board.shm")
        self.clock = _Clock()

    def tearDown(self):
        self.dir.cleanup()

    def _writer(self, role, slot, rank=0):
        return HeartbeatWriter(
            self.path, role=role, slot=slot, rank=rank, clock=self.clock
        )

    def test_stale_slot_is_reported_by_role_rank_and_age(self):
        """A rank that stops beating must surface; the ones still beating must not."""
        detok = self._writer(ROLE_DETOKENIZER, DETOKENIZER_SLOT)
        rank0 = self._writer(ROLE_SCHEDULER, 1, rank=0)
        rank3 = self._writer(ROLE_SCHEDULER, 4, rank=3)
        reader = HeartbeatReader(self.path, clock=self.clock)
        self.clock.t += 45
        detok.beat()
        rank0.beat()
        # rank 3 is stuck.
        self.assertEqual(
            reader.stale(30.0),
            [HeartbeatSlot(role="scheduler", rank=3, age_s=45.0, paused=False)],
        )
        self.assertEqual(len(reader.read()), 3)
        rank3.beat()
        self.assertEqual(reader.stale(30.0), [])

    def test_paused_slot_is_never_stale(self):
        """A compile or a known-long step pauses the clock for that rank only."""
        rank0 = self._writer(ROLE_SCHEDULER, 1)
        rank1 = self._writer(ROLE_SCHEDULER, 2, rank=1)
        rank0.set_paused(True)
        self.clock.t += 600
        reader = HeartbeatReader(self.path, clock=self.clock)
        self.assertEqual([s.rank for s in reader.stale(60.0)], [1])
        rank0.set_paused(False)
        self.clock.t += 61
        self.assertEqual(sorted(s.rank for s in reader.stale(60.0)), [0, 1])

    def test_reader_without_a_board_reports_nothing(self):
        reader = HeartbeatReader(os.path.join(self.dir.name, "missing.shm"))
        self.assertEqual(reader.read(), [])
        self.assertEqual(reader.stale(1.0), [])

    def test_never_written_slots_are_ignored(self):
        self._writer(ROLE_SCHEDULER, 7, rank=6)
        reader = HeartbeatReader(self.path, clock=self.clock)
        self.assertEqual([(s.role, s.rank) for s in reader.read()], [("scheduler", 6)])

    def test_slot_outside_capacity_is_refused(self):
        with self.assertRaises(ValueError):
            self._writer(ROLE_SCHEDULER, SLOT_CAPACITY)

    def test_scheduler_slots_are_unique_per_rank(self):
        """dp x pp x tp ranks must never share a slot, and never take slot 0."""
        slots = [
            scheduler_slot(dp_rank=dp, pp_rank=pp, tp_rank=tp, pp_size=2, tp_size=8)
            for dp in range(2)
            for pp in range(2)
            for tp in range(8)
        ]
        self.assertEqual(len(set(slots)), 32)
        self.assertNotIn(DETOKENIZER_SLOT, slots)
        self.assertEqual(min(slots), 1)

    def test_path_is_stable_per_instance_and_distinct_across_instances(self):
        self.assertEqual(heartbeat_shm_path("abc123"), heartbeat_shm_path("abc123"))
        self.assertNotEqual(heartbeat_shm_path("abc123"), heartbeat_shm_path("abc124"))
        self.assertTrue(
            heartbeat_shm_path("a/b c").startswith("/dev/shm/sglang_heartbeat_")
        )

    def test_factories_honour_the_kill_switch_and_missing_instance(self):
        with envs.SGLANG_ENABLE_HEALTH_HEARTBEAT.override(False):
            self.assertIsNone(
                create_heartbeat_writer("x", role=ROLE_SCHEDULER, slot=1, rank=0)
            )
            self.assertIsNone(create_heartbeat_reader("x"))
        self.assertIsNone(
            create_heartbeat_writer("", role=ROLE_SCHEDULER, slot=1, rank=0)
        )
        self.assertIsNone(create_heartbeat_reader(""))


class TestHealthEndpointLiveness(CustomTestCase):
    def _tm(self, reader):
        tm = TokenizerManager.__new__(TokenizerManager)
        tm.gracefully_exit = False
        tm.server_status = ServerStatus.Up
        tm.heartbeat_reader = reader
        return tm

    def _call(self, tm, path="/health"):
        prior_state = http_server.get_global_state()
        http_server.set_global_state(SimpleNamespace(tokenizer_manager=tm))
        try:
            request = SimpleNamespace(url=SimpleNamespace(path=path))
            with envs.SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION.override(False):
                return asyncio.run(http_server.health_generate(request))
        finally:
            http_server._global_state = prior_state

    def test_stale_heartbeat_turns_health_into_503_with_the_culprit_named(self):
        stale = [HeartbeatSlot(role="scheduler", rank=3, age_s=61.0, paused=False)]
        reader = SimpleNamespace(stale=lambda max_age: stale)
        response = self._call(self._tm(reader))
        self.assertEqual(response.status_code, 503)
        body = msgspec.json.decode(response.body)
        self.assertEqual(body["reason"], "stale_heartbeat")
        self.assertEqual(body["stale"][0]["rank"], 3)
        self.assertEqual(body["stale"][0]["role"], "scheduler")

    def test_fresh_heartbeats_and_no_reader_keep_health_200(self):
        self.assertEqual(
            self._call(self._tm(SimpleNamespace(stale=lambda max_age: []))).status_code,
            200,
        )
        self.assertEqual(self._call(self._tm(None)).status_code, 200)

    def test_max_age_comes_from_the_env(self):
        seen = []

        def stale(max_age):
            seen.append(max_age)
            return []

        with envs.SGLANG_HEALTH_HEARTBEAT_MAX_AGE_S.override(12.5):
            self._call(self._tm(SimpleNamespace(stale=stale)))
        self.assertEqual(seen, [12.5])


if __name__ == "__main__":
    unittest.main()
