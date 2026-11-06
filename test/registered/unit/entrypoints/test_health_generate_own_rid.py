"""/health_generate is answered only by the probe's own request.

Before this change the handler returned 200 as soon as the tokenizer manager
received *any* message from the scheduler or detokenizer, including another
request's failure notification. These tests pin the new contract: the probe
resolves on its own first output, on the scheduler's HealthCheckOutput carrying
its rid (the busy shortcut), or on its own failure; nothing else moves it.

Usage:
    python3 -m pytest test/registered/unit/entrypoints/test_health_generate_own_rid.py -v
"""

import asyncio
import unittest
from collections import deque
from types import SimpleNamespace
from unittest import mock

from sglang.srt.entrypoints import http_server
from sglang.srt.managers.io_struct import HealthCheckOutput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import (
    HealthCheckWaiter,
    ServerStatus,
    TokenizerManager,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

PROBE_TIMEOUT_S = 0.3


class _FakeRequest:
    url = SimpleNamespace(path="/health_generate")


def _make_tokenizer_manager(generate_request):
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.gracefully_exit = False
    tm.server_status = ServerStatus.Up
    tm.is_generation = True
    tm.rid_to_state = {}
    tm.health_check_waiters = {}
    tm.last_receive_tstamp = 0.0
    tm.generate_request = generate_request
    return tm


class TestHealthGenerateOwnRid(CustomTestCase):
    def run_probe(self, generate_request, side_task=None):
        """Run the handler against a fake tokenizer manager; return (response, tm, rids)."""
        tm = _make_tokenizer_manager(generate_request)
        prior_state = http_server.get_global_state()
        http_server.set_global_state(SimpleNamespace(tokenizer_manager=tm))
        patches = [
            mock.patch.object(http_server, "HEALTH_CHECK_TIMEOUT", PROBE_TIMEOUT_S),
            mock.patch.object(
                http_server,
                "get_disagg",
                lambda: SimpleNamespace(disaggregation_mode="null"),
            ),
        ]
        for p in patches:
            p.start()
        try:

            async def run():
                task = asyncio.create_task(http_server.health_generate(_FakeRequest()))
                if side_task is not None:
                    await side_task(tm)
                return await task

            return asyncio.run(run()), tm
        finally:
            for p in patches:
                p.stop()
            http_server._global_state = prior_state

    def test_other_requests_message_does_not_satisfy_the_probe(self):
        """The old contract: any message bumps last_receive_tstamp -> 200. Now: 503."""
        seen = []

        async def never_answers(gri, request):
            seen.append(gri.rid)
            tm = http_server.get_global_state().tokenizer_manager
            # Another request's output arrives; the old handler returned 200 here.
            tm.last_receive_tstamp = 1e12
            await asyncio.sleep(10)
            yield {}  # pragma: no cover - cancelled before this

        response, tm = self.run_probe(never_answers)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(tm.server_status, ServerStatus.UnHealthy)
        self.assertEqual(tm.health_check_waiters, {})
        self.assertNotIn(seen[0], tm.rid_to_state)

    def test_own_first_output_returns_200(self):
        async def answers(gri, request):
            await asyncio.sleep(0.02)
            yield {"text": ""}

        response, tm = self.run_probe(answers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(tm.server_status, ServerStatus.Up)
        self.assertEqual(tm.health_check_waiters, {})

    def test_scheduler_health_check_output_for_own_rid_returns_200(self):
        """Busy engine: the scheduler answers the probe by rid, no output is produced."""
        probe_rid = {}

        async def never_answers(gri, request):
            probe_rid["rid"] = gri.rid
            await asyncio.sleep(10)
            yield {}  # pragma: no cover

        async def scheduler_answers(tm):
            await asyncio.sleep(0.02)
            TokenizerManager._handle_health_check_output(
                tm, HealthCheckOutput(rid=probe_rid["rid"], http_worker_ipc=None)
            )

        response, tm = self.run_probe(never_answers, side_task=scheduler_answers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(tm.server_status, ServerStatus.Up)

    def test_scheduler_health_check_output_for_another_rid_is_ignored(self):
        async def never_answers(gri, request):
            await asyncio.sleep(10)
            yield {}  # pragma: no cover

        async def wrong_rid(tm):
            await asyncio.sleep(0.02)
            TokenizerManager._handle_health_check_output(
                tm,
                HealthCheckOutput(
                    rid="HEALTH_CHECK_someone_else", http_worker_ipc=None
                ),
            )
            # A legacy signal without a rid must not count either.
            TokenizerManager._handle_health_check_output(
                tm, HealthCheckOutput(rid=None, http_worker_ipc=None)
            )

        response, tm = self.run_probe(never_answers, side_task=wrong_rid)
        self.assertEqual(response.status_code, 503)

    def test_probe_failure_returns_503_before_the_timeout(self):
        async def fails(gri, request):
            await asyncio.sleep(0.01)
            raise ValueError("Decode transfer failed for the probe")
            yield {}  # pragma: no cover

        response, tm = self.run_probe(fails)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(tm.server_status, ServerStatus.UnHealthy)
        self.assertEqual(tm.health_check_waiters, {})

    def test_waiter_resolves_once(self):
        waiter = HealthCheckWaiter(event=asyncio.Event())
        waiter.resolve(False)
        waiter.resolve(True)
        self.assertTrue(waiter.event.is_set())
        self.assertFalse(waiter.ok)


class TestRequestStateStillConstructs(unittest.TestCase):
    def test_req_state_keeps_its_dataclass_constructor(self):
        """Regression: inserting HealthCheckWaiter above ReqState must not steal its decorator."""
        import dataclasses

        from sglang.srt.managers.tokenizer_manager import ReqState

        self.assertTrue(dataclasses.is_dataclass(ReqState))
        fields = {f.name for f in dataclasses.fields(ReqState)}
        self.assertTrue({"out_list", "finished", "event", "obj"} <= fields)


class TestSchedulerDeferredProbeCarriesRid(unittest.TestCase):
    def test_maybe_send_health_check_signal_sends_the_probes_rid(self):
        sent = []
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.return_health_check_ipcs = deque(
            [("HEALTH_CHECK_abc", "ipc://worker-1"), ("HEALTH_CHECK_def", None)]
        )
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=sent.append)
        )
        scheduler.maybe_send_health_check_signal()
        scheduler.maybe_send_health_check_signal()
        scheduler.maybe_send_health_check_signal()  # empty deque: no send
        self.assertEqual(len(sent), 2)
        self.assertIsInstance(sent[0], HealthCheckOutput)
        self.assertEqual(
            [(o.rid, o.http_worker_ipc) for o in sent],
            [("HEALTH_CHECK_abc", "ipc://worker-1"), ("HEALTH_CHECK_def", None)],
        )


if __name__ == "__main__":
    unittest.main()
