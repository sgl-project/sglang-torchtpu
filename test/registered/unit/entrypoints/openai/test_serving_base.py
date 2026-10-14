"""Unit tests for OpenAIServingBase.handle_request request conversion.

Conversion (chat template rendering + prompt tokenization) runs off the event
loop, so a slow conversion must not stall other coroutines such as /health,
and its errors must still map to the same HTTP responses (issue #1336).
"""

import asyncio
import json
import threading
import unittest
from types import SimpleNamespace

from fastapi import HTTPException

from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Upper bound for every wait below; a pass never waits for it.
_TIMEOUT_S = 5.0


class _ScriptedServing(OpenAIServingBase):
    """Handler whose conversion blocks until released, then runs ``convert``."""

    def __init__(self, convert=None):
        super().__init__(
            SimpleNamespace(
                request_logger=SimpleNamespace(log_requests=False, log_requests_level=0)
            )
        )
        self.convert = convert or (lambda request: (request, request))
        self.conversion_started = threading.Event()
        self.release_conversion = threading.Event()
        self.conversion_thread = None

    def _request_id_prefix(self) -> str:
        return "test-"

    def _convert_to_internal_request(self, request, raw_request=None):
        self.conversion_thread = threading.current_thread()
        self.conversion_started.set()
        if not self.release_conversion.wait(timeout=_TIMEOUT_S):
            # Only reachable when conversion blocks the loop that would release it.
            raise RuntimeError("conversion was never released")
        return self.convert(request)

    async def _handle_non_streaming_request(
        self, adapted_request, request, raw_request
    ):
        return "converted"


class TestServingBaseRequestConversion(unittest.IsolatedAsyncioTestCase):
    async def test_blocked_conversion_does_not_stall_the_event_loop(self):
        serving = _ScriptedServing()
        loop_thread = threading.current_thread()

        request_task = asyncio.create_task(
            serving.handle_request(SimpleNamespace(stream=False), None)
        )
        self.assertTrue(
            await asyncio.to_thread(serving.conversion_started.wait, _TIMEOUT_S)
        )

        # A concurrent coroutine (e.g. a /health probe) completes while the
        # conversion is still blocked.
        probe = asyncio.create_task(asyncio.sleep(0, result="healthy"))
        self.assertEqual(await asyncio.wait_for(probe, _TIMEOUT_S), "healthy")
        self.assertFalse(request_task.done())

        serving.release_conversion.set()
        response = await asyncio.wait_for(request_task, _TIMEOUT_S)
        self.assertEqual(response, "converted")
        self.assertIsNot(serving.conversion_thread, loop_thread)

    async def test_conversion_errors_keep_their_http_mapping(self):
        def raise_(exc):
            def convert(request):
                raise exc

            return convert

        cases = [
            (ValueError("bad prompt"), 400, "BadRequest", "bad prompt"),
            (
                HTTPException(status_code=422, detail="bad header"),
                422,
                "422",
                "bad header",
            ),
            (
                RuntimeError("boom"),
                500,
                "InternalServerError",
                "Internal server error: boom",
            ),
        ]
        for exc, status_code, err_type, message in cases:
            with self.subTest(exc=type(exc).__name__):
                serving = _ScriptedServing(convert=raise_(exc))
                serving.release_conversion.set()

                response = await asyncio.wait_for(
                    serving.handle_request(SimpleNamespace(stream=False), None),
                    _TIMEOUT_S,
                )

                self.assertEqual(response.status_code, status_code)
                body = json.loads(response.body)
                self.assertEqual(body["type"], err_type)
                self.assertEqual(body["message"], message)
                self.assertIsNot(serving.conversion_thread, threading.current_thread())


if __name__ == "__main__":
    unittest.main()
