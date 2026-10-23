import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.http_server import (
    _send_disaggregation_warmup_requests,
)
from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.srt.managers.io_struct import FreezeGCReq, FreezeGCReqOutput
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
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


class TestFreezeGcAfterWarmup(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.server_args = SimpleNamespace()
        self.manager = SimpleNamespace(server_status=None, freeze_gc=AsyncMock())
        stubs = dict(
            get_model=lambda: SimpleNamespace(
                checkpoint_engine_wait_weights_before_ready=False,
                delete_ckpt_after_loading=False,
            ),
            get_exec=lambda: SimpleNamespace(
                moe=SimpleNamespace(is_ep_scale_joiner=False)
            ),
            get_serving=lambda: SimpleNamespace(skip_server_warmup=True),
            get_observability=lambda: SimpleNamespace(
                debug_tensor_dump_input_file=None
            ),
            _global_state=SimpleNamespace(tokenizer_manager=self.manager),
        )
        for name, value in stubs.items():
            p = patch.object(http_server, name, value)
            p.start()
            self.addCleanup(p.stop)

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


class TestPipelineFreeze(unittest.TestCase):
    def test_deferred_dispatch_only_handles_freeze(self):
        processed = []
        scheduler = SimpleNamespace(process_input_requests=processed.extend)
        request = FreezeGCReq()
        SchedulerPPMixin._pp_process_freeze_gc(scheduler, [object(), request])
        self.assertEqual(processed, [request])


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
