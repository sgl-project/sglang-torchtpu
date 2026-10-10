import asyncio
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.http_server import (
    _send_disaggregation_warmup_requests,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestDisaggregationServerWarmup(unittest.IsolatedAsyncioTestCase):
    async def test_sends_concurrent_scalar_request_to_each_dp_rank(self):
        from sglang.srt.runtime_context import get_context

        # The warmup fan-out width comes from the published topology.
        override = get_context().override_server_args(dp_size=4)
        override.install()
        self.addCleanup(override.restore)
        server_args = SimpleNamespace(dp_size=4)
        all_started = asyncio.Event()
        calls = []
        sessions = []

        class Response:
            status = 200

            async def __aenter__(self):
                if len(calls) == server_args.dp_size:
                    all_started.set()
                await asyncio.wait_for(all_started.wait(), timeout=5)
                return self

            async def __aexit__(self, *args):
                pass

            async def read(self):
                return b""

        class Session:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                sessions.append(self)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def post(self, *args, **kwargs):
                calls.append((args, kwargs))
                return Response()

        with patch("sglang.srt.entrypoints.http_server.aiohttp.ClientSession", Session):
            status_codes = await _send_disaggregation_warmup_requests(
                url="http://localhost:30000",
                headers={"Authorization": "Bearer token"},
                ssl_verify=False,
                timeout=123,
            )

        self.assertEqual(status_codes, [200] * server_args.dp_size)
        self.assertEqual(len(calls), server_args.dp_size)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(
            sessions[0].kwargs["headers"], {"Authorization": "Bearer token"}
        )
        self.assertEqual(sessions[0].kwargs["timeout"].total, 123)

        calls_by_rank = {
            kwargs["json"]["routed_dp_rank"]: (args, kwargs) for args, kwargs in calls
        }
        self.assertEqual(set(calls_by_rank), set(range(server_args.dp_size)))

        for dp_rank, (args, kwargs) in calls_by_rank.items():
            self.assertEqual(args, ("http://localhost:30000/generate",))
            self.assertEqual(kwargs["json"]["input_ids"], [10, 11, 12, 13])
            self.assertEqual(kwargs["json"]["bootstrap_host"], FAKE_BOOTSTRAP_HOST)
            self.assertEqual(kwargs["json"]["bootstrap_room"], dp_rank)
            self.assertFalse(kwargs["ssl"])


class _FreezeOk:
    def raise_for_status(self):
        return None


class _FreezeServerError:
    def raise_for_status(self):
        raise requests.exceptions.HTTPError("500")


class TestFreezeGcAfterWarmup(unittest.TestCase):
    def setUp(self):
        self.server_args = SimpleNamespace(url=lambda: "http://127.0.0.1:30000")
        self.calls = []
        self.sleeps = []
        # Every clock read advances one second.
        clock = itertools.count()
        stubs = dict(
            get_model=lambda: SimpleNamespace(
                checkpoint_engine_wait_weights_before_ready=False,
                delete_ckpt_after_loading=False,
            ),
            get_exec=lambda: SimpleNamespace(
                moe=SimpleNamespace(is_ep_scale_joiner=False)
            ),
            get_serving=lambda: SimpleNamespace(
                skip_server_warmup=True, admin_api_key=None, api_key="k"
            ),
            get_observability=lambda: SimpleNamespace(
                debug_tensor_dump_input_file=None
            ),
            ssl_verify_of=lambda args: True,
            _global_state=SimpleNamespace(
                tokenizer_manager=SimpleNamespace(server_status=None)
            ),
            time=SimpleNamespace(
                monotonic=lambda: float(next(clock)), sleep=self.sleeps.append
            ),
        )
        for name, value in stubs.items():
            p = patch.object(http_server, name, value)
            p.start()
            self.addCleanup(p.stop)

    def _post(self, outcomes):
        outcomes = list(outcomes)

        def post(url, **kwargs):
            self.calls.append((url, kwargs))
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        return patch.object(http_server.requests, "post", post)

    def test_skip_server_warmup_freezes_once_the_server_listens(self):
        """With --skip-server-warmup, /freeze_gc can run before the HTTP server
        listens; the heap must still be frozen before the server reports ready."""
        refused = requests.exceptions.ConnectionError("Connection refused")
        ready = []
        with self._post([refused, refused, refused, _FreezeOk()]):
            http_server._wait_and_warmup(
                self.server_args, launch_callback=lambda: ready.append(len(self.calls))
            )
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(ready, [4])

    def test_retry_is_bounded(self):
        refused = requests.exceptions.ConnectionError("Connection refused")
        with (
            patch.object(http_server, "_FREEZE_GC_RETRY_SECS", 1.5),
            self._post([refused] * 3),
        ):
            http_server._freeze_gc_after_server_warmup(self.server_args)
        self.assertEqual(len(self.calls), 2)

    def test_http_and_ssl_errors_are_not_retried(self):
        for error in (
            _FreezeServerError(),
            requests.exceptions.SSLError("bad certificate"),
        ):
            with self.subTest(error=type(error).__name__):
                self.calls.clear()
                with self._post([error, _FreezeOk()]):
                    http_server._freeze_gc_after_server_warmup(self.server_args)
                self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
