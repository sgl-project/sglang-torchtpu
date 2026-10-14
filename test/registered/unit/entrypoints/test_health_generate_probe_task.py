"""/health_generate must not leave its abandoned probe's exception unretrieved.

health_generate answers from last_receive_tstamp and only requests cancellation
of its probe task. Near the probe's own wait_for timeout in
_stream_one_response the cancel can lose: the task then ends with the
client-disconnect ValueError (the handler has already returned). Nothing awaits
the task, so asyncio used to log "Task exception was never retrieved" (#1327).
"""

import asyncio
import gc
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints import http_server
from sglang.srt.managers.tokenizer_manager import ServerStatus, TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

DISCONNECT_ERROR = "Request is disconnected from the client side (type 1)."


def _probe_that_loses_the_cancel():
    # Deterministic stand-in for the race: the cancel arrives as
    # _stream_one_response's wait_for times out, and the task finishes with the
    # client-disconnect error instead of CancelledError.
    async def generate_request(obj, request=None):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise ValueError(DISCONNECT_ERROR)
        yield

    return generate_request


def _probe_that_honors_the_cancel():
    async def generate_request(obj, request=None):
        await asyncio.Event().wait()
        yield

    return generate_request


class TestHealthGenerateProbeTask(CustomTestCase):
    def setUp(self):
        super().setUp()
        enter_scope(self, published_topology(role="tokenizer"))

    def call_health_generate(self, generate_request, *, last_receive_tstamp):
        tokenizer_manager = TokenizerManager.__new__(TokenizerManager)
        tokenizer_manager.gracefully_exit = False
        tokenizer_manager.server_status = ServerStatus.Up
        tokenizer_manager.is_generation = True
        tokenizer_manager.last_receive_tstamp = last_receive_tstamp
        tokenizer_manager.rid_to_state = {}
        tokenizer_manager.generate_request = generate_request
        request = SimpleNamespace(url=SimpleNamespace(path="/health_generate"))
        unretrieved = []

        async def run():
            loop = asyncio.get_running_loop()
            loop.set_exception_handler(
                lambda _loop, context: unretrieved.append(context["message"])
            )
            response = await http_server.health_generate(request)
            # Let the abandoned probe finish, then collect it: asyncio reports an
            # unretrieved exception when the finished task is destroyed.
            for _ in range(5):
                await asyncio.sleep(0)
            gc.collect()
            return response

        prior_state = http_server.get_global_state()
        http_server.set_global_state(
            SimpleNamespace(tokenizer_manager=tokenizer_manager)
        )
        try:
            response = asyncio.run(run())
        finally:
            http_server._global_state = prior_state
        return response, tokenizer_manager, unretrieved

    def test_healthy_probe_whose_cancel_is_lost_logs_nothing(self):
        with self.assertLogs(http_server.logger, level="DEBUG") as logs:
            # assertLogs requires one record; keep the evidence below in order.
            http_server.logger.debug("start")
            response, tokenizer_manager, unretrieved = self.call_health_generate(
                _probe_that_loses_the_cancel(), last_receive_tstamp=float("inf")
            )

        self.assertEqual(unretrieved, [])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(tokenizer_manager.server_status, ServerStatus.Up)
        self.assertTrue(
            any(
                "Abandoned health check probe ended with ValueError" in line
                and DISCONNECT_ERROR in line
                for line in logs.output
            ),
            logs.output,
        )

    def test_timed_out_probe_whose_cancel_is_lost_logs_nothing(self):
        with patch.object(http_server, "HEALTH_CHECK_TIMEOUT", 0.5):
            response, tokenizer_manager, unretrieved = self.call_health_generate(
                _probe_that_loses_the_cancel(), last_receive_tstamp=0.0
            )

        self.assertEqual(unretrieved, [])
        self.assertEqual(response.status_code, 503)
        self.assertEqual(tokenizer_manager.server_status, ServerStatus.UnHealthy)

    def test_cancelled_probe_is_silent(self):
        with self.assertNoLogs(http_server.logger, level="DEBUG"):
            response, _, unretrieved = self.call_health_generate(
                _probe_that_honors_the_cancel(), last_receive_tstamp=time.time() + 60
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(unretrieved, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
