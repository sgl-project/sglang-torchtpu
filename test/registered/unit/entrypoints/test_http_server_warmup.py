import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.http_server import (
    _send_disaggregation_warmup_requests,
)
from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.srt.managers.io_struct import FreezeGCReqOutput
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import enter_scope

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestDisaggregationServerWarmup(unittest.IsolatedAsyncioTestCase):
    async def test_sends_concurrent_scalar_request_to_each_dp_rank(self):
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


class TestFreezeGcAfterWarmup(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.server_args = enter_scope(
            self,
            get_context().override_server_args(
                host="127.0.0.1", port=30000, skip_server_warmup=True, api_key="k"
            ),
        )
        self.manager = SimpleNamespace(server_status=None, freeze_gc=AsyncMock())
        enter_scope(
            self,
            patch.object(
                http_server,
                "_global_state",
                http_server._GlobalState(
                    tokenizer_manager=self.manager,
                    template_manager=None,
                    scheduler_info={},
                ),
            ),
        )

    async def test_skip_warmup_waits_for_cleanup_on_serving_loop(self):
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        finish = asyncio.Event()
        ready = []

        async def cleanup():
            self.assertIs(asyncio.get_running_loop(), loop)
            started.set()
            await finish.wait()

        self.manager.freeze_gc.side_effect = cleanup
        task = asyncio.create_task(
            asyncio.to_thread(
                http_server._wait_and_warmup,
                self.server_args,
                launch_callback=lambda: ready.append(True),
                event_loop=loop,
            )
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            self.assertEqual(ready, [])
            self.assertIsNone(self.manager.server_status)
        finally:
            finish.set()
            await asyncio.wait_for(task, timeout=5)
        self.manager.freeze_gc.assert_awaited_once()
        self.assertEqual(ready, [True])
        self.assertEqual(self.manager.server_status, http_server.ServerStatus.Up)

    async def test_cleanup_failure_does_not_publish_readiness(self):
        self.manager.freeze_gc.side_effect = RuntimeError("cleanup failed")
        ready = []
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            await asyncio.to_thread(
                http_server._wait_and_warmup,
                self.server_args,
                launch_callback=lambda: ready.append(True),
                event_loop=asyncio.get_running_loop(),
            )
        self.assertEqual(ready, [])
        self.assertIsNone(self.manager.server_status)

    async def test_rust_server_keeps_its_own_readiness(self):
        ready = []
        with http_server.envs.SGLANG_RUST_SERVER.override(True):
            await asyncio.to_thread(
                http_server._wait_and_warmup,
                self.server_args,
                launch_callback=lambda: ready.append(True),
                event_loop=asyncio.get_running_loop(),
            )
        self.manager.freeze_gc.assert_not_called()
        self.assertEqual(ready, [True])
        self.assertIsNone(self.manager.server_status)

    async def test_failed_pd_warmup_does_not_publish_readiness(self):
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                manager = http_server._global_state.tokenizer_manager
                manager.server_status = http_server.ServerStatus.Starting
                ready = Mock()
                with (
                    get_context().override_server_args(
                        skip_server_warmup=False,
                        skip_tokenizer_init=True,
                        api_key="k",
                        disaggregation_mode=mode,
                        dp_size=2,
                    ),
                    patch.object(
                        http_server.requests,
                        "get",
                        return_value=SimpleNamespace(
                            status_code=200, json=lambda: {"is_generation": True}
                        ),
                    ),
                    patch.object(
                        http_server,
                        "_send_disaggregation_warmup_requests",
                        new_callable=AsyncMock,
                        return_value=[200, 503],
                    ) as warmup,
                    patch.object(http_server, "kill_process_tree") as shutdown,
                ):
                    await asyncio.to_thread(
                        http_server._wait_and_warmup,
                        self.server_args,
                        launch_callback=ready,
                        event_loop=asyncio.get_running_loop(),
                    )
                warmup.assert_awaited_once()
                shutdown.assert_called_once_with(http_server.os.getpid())
                self.manager.freeze_gc.assert_not_called()
                ready.assert_not_called()
                self.assertEqual(
                    manager.server_status, http_server.ServerStatus.Starting
                )


class TestFreezeCompletion(unittest.IsolatedAsyncioTestCase):
    async def test_waits_for_every_scheduler_reply_before_local_freeze(self):
        events = []
        communicator = FanOutCommunicator(lambda _: events.append("sent"), 2)
        manager = SimpleNamespace(
            auto_create_handle_loop=lambda: None,
            freeze_gc_communicator=communicator,
        )
        with patch(
            "sglang.srt.managers.tokenizer_manager.freeze_gc", side_effect=events.append
        ):
            task = asyncio.create_task(http_server.TokenizerManager.freeze_gc(manager))
            await asyncio.sleep(0)
            communicator.handle_recv(FreezeGCReqOutput())
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            self.assertEqual(events, ["sent"])
            communicator.handle_recv(FreezeGCReqOutput())
            await task
        self.assertEqual(events, ["sent", "Tokenizer Manager"])


if __name__ == "__main__":
    unittest.main()
