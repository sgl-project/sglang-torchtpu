import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints import http_server
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeTokenizerManager:
    """Records whether each abort could still resolve its rid.

    `abort_request` looks the rid up in `rid_to_state` and drops the abort when
    it is missing, so aborting after the pop is silently a no-op.
    """

    def __init__(self, respond: bool, finished: bool = False):
        self.is_generation = True
        self.gracefully_exit = False
        self.rid_to_state = {}
        self.server_status = None
        self.last_receive_tstamp = float("inf") if respond else 0.0
        self.finished = finished
        self.aborted = []

    async def generate_request(self, obj, request):
        self.rid_to_state[obj.rid] = SimpleNamespace(finished=self.finished)
        yield {}

    def abort_request(self, rid: str = "", abort_all: bool = False):
        self.aborted.append((rid, rid in self.rid_to_state))


class TestHealthGenerateProbeRelease(unittest.IsolatedAsyncioTestCase):
    async def _run(self, respond: bool, finished: bool = False):
        tm = FakeTokenizerManager(respond, finished)
        request = SimpleNamespace(url=SimpleNamespace(path="/health_generate"))
        # Decode mode is the case that matters: the probe then carries a fake
        # bootstrap peer and occupies a decode prealloc queue slot.
        disagg = SimpleNamespace(
            disaggregation_mode=http_server.DisaggregationMode.DECODE.value
        )
        with (
            patch.object(
                http_server, "_global_state", SimpleNamespace(tokenizer_manager=tm)
            ),
            patch.object(http_server, "HEALTH_CHECK_TIMEOUT", 2),
            patch.object(http_server, "get_disagg", lambda: disagg),
        ):
            await http_server.health_generate(request)
        return tm

    async def test_unfinished_probe_is_aborted_while_its_rid_still_resolves(self):
        # The success path needs this as much as the timeout path: it returns on
        # any response reaching the manager, not necessarily this probe's.
        for respond in (True, False):
            with self.subTest(responded=respond):
                tm = await self._run(respond)
                self.assertEqual(len(tm.aborted), 1)
                rid, resolvable = tm.aborted[0]
                self.assertTrue(rid.startswith("HEALTH_CHECK"))
                self.assertTrue(resolvable, "abort ran after the rid was popped")
                self.assertNotIn(rid, tm.rid_to_state)

    async def test_finished_probe_is_released_without_an_abort(self):
        # Nothing to reclaim, so it must not land in the aborted-request metric.
        tm = await self._run(respond=True, finished=True)
        self.assertEqual(tm.aborted, [])
        self.assertEqual(tm.rid_to_state, {})


if __name__ == "__main__":
    unittest.main()
