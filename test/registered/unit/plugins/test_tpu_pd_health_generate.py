"""P/D health: completed chunks or tokens keep work healthy; stalls return 503.

CPU tests use real handlers and IPC types with controlled time and transport.
"""

import asyncio
import sys
import time
from array import array
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import msgspec
import pytest
from fastapi import FastAPI, Request, Response
from fastapi.routing import APIRoute

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.entrypoints import http_server
from sglang.srt.environ import envs
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers import tokenizer_manager as tm_module
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchStrOutput,
    BatchTokenIDOutput,
    GenerateReqInput,
    HealthCheckOutput,
    PauseContinueBroadcastReq,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import (
    ReqState,
    ServerStatus,
    TokenizerManager,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability import req_time_stats
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.utils.watchdog import SubprocessWatchdog
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import published_topology
from sglang.utils import TypeBasedDispatcher
from sglang_tpu import patches
from sglang_tpu.disaggregation import health_generate as hg
from sglang_tpu.envs import envs as tpu_envs

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

# Async cases run on anyio's pytest plugin (anyio is a core dependency); no
# pytest-asyncio in the CPU gate.
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


WINDOW = 100.0
WINDOW_ENV = tpu_envs.SGLANG_TPU_PD_HEALTH_WINDOW_S


@dataclass
class Clock:
    monotonic: float = 10000.0
    wall: float = 1700000000.0

    def perf_counter(self):
        return self.monotonic

    def time(self):
        return self.wall

    def advance(self, seconds):
        self.monotonic += seconds
        self.wall += seconds


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    # Replace module references, not attributes of the shared time module:
    # asyncio and the test runner must retain their real clocks.
    monkeypatch.setattr(hg, "time", clock)
    monkeypatch.setattr(hg, "real_time", clock.time)
    monkeypatch.setattr(req_time_stats, "time", clock)
    monkeypatch.setattr(tm_module, "time", clock)
    return clock


@pytest.fixture
def tm(clock):
    with published_topology(
        role="tokenizer",
        disaggregation_mode="decode",
        batch_notify_size=1,
        speculative_algorithm=None,
    ) as args:
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.server_args = args
        manager._config_updates = []
        manager.rid_to_state = {}
        manager.server_status = ServerStatus.Up
        manager.gracefully_exit = False
        manager._is_pause = False
        manager.last_receive_tstamp = clock.time()
        manager.disaggregation_mode = DisaggregationMode.DECODE
        manager.enable_metrics = False
        manager.enable_priority_scheduling = False
        manager.metrics_collector = Mock(labels={})
        manager.enable_lora = False
        manager.incremental_streaming_output = False
        manager.skip_tokenizer_init = False
        manager.dump_requests_folder = ""
        manager.crash_dump_folder = ""
        yield manager


def state(clock, rid="request", *, age=0, produced=None, dispatched=True, **kwargs):
    obj = GenerateReqInput(
        rid=rid, input_ids=[10], sampling_params={}, stream=True, **kwargs
    )
    stats = APIServerReqTimeStats(
        created_time=clock.perf_counter() - age - 1,
        api_server_dispatch_time=clock.perf_counter() - age,
    )
    if produced is not None:
        stats.set_first_token_time(clock.perf_counter() - produced)
    return ReqState(
        out_list=[],
        finished=False,
        event=asyncio.Event(),
        obj=obj,
        time_stats=stats,
        dispatched=dispatched,
    )


def batch(rid, *, finished=None, tokens=1, kind="text"):
    """Real output structures; optional tensors/logprobs are deliberately absent."""
    common = dict(
        rids=[rid],
        finished_reasons=[finished],
        output_ids=[array("i", [10])],
        prompt_tokens=[1],
        completion_tokens=[tokens],
        reasoning_tokens=[0],
        cached_tokens=[0],
        retraction_counts=[0],
        input_token_logprobs_val=None,
        input_token_logprobs_idx=None,
        output_token_logprobs_val=None,
        output_token_logprobs_idx=None,
        input_top_logprobs_val=None,
        input_top_logprobs_idx=None,
        output_top_logprobs_val=None,
        output_top_logprobs_idx=None,
        input_token_ids_logprobs_val=None,
        input_token_ids_logprobs_idx=None,
        output_token_ids_logprobs_val=None,
        output_token_ids_logprobs_idx=None,
        output_token_entropy_val=None,
        output_token_sampling_mask=None,
        output_hidden_states=None,
        routed_experts=None,
        indexer_topk=None,
        placeholder_tokens_idx=None,
        placeholder_tokens_val=None,
    )
    if kind == "text":
        return BatchStrOutput(output_strs=["a"], **common)
    return BatchTokenIDOutput(
        decoded_texts=[""],
        decode_ids=[array("i", [10])],
        read_offsets=[0],
        skip_special_tokens=[True],
        spaces_between_special_tokens=[True],
        no_stop_trim=[False],
        **common,
    )


def request(path="/health_generate"):
    return Request(
        dict(type="http", method="GET", path=path, headers=[], query_string=b"")
    )


@pytest.fixture
def route(tm, monkeypatch):
    monkeypatch.setattr(
        http_server, "_global_state", SimpleNamespace(tokenizer_manager=tm)
    )

    async def healthy_core(request):
        if tm.gracefully_exit or tm.server_status == ServerStatus.Starting:
            return Response(status_code=503)
        return Response(status_code=200)

    core = AsyncMock(side_effect=healthy_core)
    monkeypatch.setattr(hg, "_ORIGINAL_HANDLER", {"handler": core})
    with WINDOW_ENV.override(WINDOW):
        yield core


@pytest.fixture
def wrapped(monkeypatch):
    # Restore every process-global mutation, including when a test fails.
    for name in (
        "_handle_batch_output",
        "init_request_dispatcher",
    ):
        monkeypatch.setattr(TokenizerManager, name, getattr(TokenizerManager, name))
    monkeypatch.setattr(patches, "_tokenizer_manager_pd_health_patched", False)
    monkeypatch.setattr(hg, "install_health_generate_route", Mock())
    patches._patch_tokenizer_manager_pd_health_generate()


@pytest.fixture
def completed_work(tm, wrapped, monkeypatch):
    monkeypatch.setattr(tm, "init_communicators", Mock())
    tm.init_request_dispatcher()
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.disaggregation_mode = DisaggregationMode.PREFILL
    scheduler.waiting_queue = []
    scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
    scheduler.ipc_channels = SimpleNamespace(
        send_to_tokenizer=SimpleNamespace(
            send_output=lambda obj: tm._result_dispatcher(
                msgpack_decode(msgpack_encode(obj))
            )
        )
    )
    for name in (
        "process_batch_result",
        "maybe_send_health_check_signal",
        "maybe_signal_queued_health_checks",
    ):
        monkeypatch.setattr(Scheduler, name, getattr(Scheduler, name))
    monkeypatch.setattr(patches, "_scheduler_pd_health_patched", False)
    patches._patch_scheduler_pd_health_progress()

    def complete(mode=ForwardMode.EXTEND):
        scheduler.maybe_signal_queued_health_checks(
            SimpleNamespace(
                reqs=[SimpleNamespace(rid="real", http_worker_ipc=None)],
                forward_mode=mode,
            )
        )

    return complete


@pytest.fixture
def native_probe(tm, clock, route, monkeypatch):
    """Run the saved core handler itself, with deterministic scheduler heartbeats."""
    driver = SimpleNamespace(acknowledge=True, probes=[], tasks=[])
    tm.is_generation = True

    async def generate(obj, raw_request):
        driver.probes.append(obj)
        driver.tasks.append(asyncio.current_task())
        tm.rid_to_state[obj.rid] = state(clock, obj.rid)
        await asyncio.Event().wait()
        yield  # Keep the actual core handler's async-iterator calling convention.

    async def tick(delay):
        assert delay == 1
        await asyncio.sleep(0)
        clock.advance(1)
        if driver.acknowledge:
            tm.last_receive_tstamp = clock.time()

    async def poll(path="/health_generate"):
        try:
            return await hg.health_generate(request(path))
        finally:
            if driver.tasks:
                await asyncio.gather(*driver.tasks, return_exceptions=True)

    driver.poll = poll
    monkeypatch.setattr(tm, "generate_request", generate)
    monkeypatch.setattr(
        hg, "_ORIGINAL_HANDLER", {"handler": http_server.health_generate}
    )
    monkeypatch.setattr(http_server, "HEALTH_CHECK_TIMEOUT", 3)
    monkeypatch.setattr(
        http_server,
        "asyncio",
        SimpleNamespace(create_task=asyncio.create_task, sleep=tick),
    )
    monkeypatch.setattr(
        http_server,
        "time",
        SimpleNamespace(
            time=clock.time, strftime=time.strftime, localtime=time.localtime
        ),
    )
    with (
        envs.SGLANG_DIAG_BYPASS_HEALTH_GENERATE.override(False),
        envs.SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION.override(False),
    ):
        yield driver


class TestRoutesAndLifecycle:
    @pytest.mark.parametrize(
        "starting,shutdown", [(True, False), (False, True), (True, True)]
    )
    async def test_lifecycle_rejections_pass_through(
        self, tm, clock, route, starting, shutdown
    ):
        tm.server_status = ServerStatus.Starting if starting else ServerStatus.Up
        tm.gracefully_exit = shutdown
        tm.rid_to_state["old"] = state(clock, age=1000)
        response = await hg.health_generate(request())
        assert response.status_code == 503
        assert response.body == b""
        assert tm.server_status == (
            ServerStatus.Starting if starting else ServerStatus.Up
        )

    async def test_real_asgi_routing_leaves_health_unchanged(
        self, tm, clock, route, monkeypatch
    ):
        app = FastAPI()

        async def core(request: Request):
            return await route(request)

        app.add_api_route("/health", core, methods=["GET"])
        app.add_api_route("/health_generate", core, methods=["GET"])
        app.add_api_route("/ready", http_server.ready, methods=["GET"])
        monkeypatch.setattr(http_server, "app", app)
        assert hg.install_health_generate_route()
        assert not hg.install_health_generate_route()
        assert (
            sum(
                isinstance(r, APIRoute) and r.path == "/health_generate"
                for r in app.routes
            )
            == 1
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get("/health_generate")).status_code == 200
            tm.rid_to_state["pending"] = state(clock, age=101)
            assert (await client.get("/health")).status_code == 200
            reply = await client.get("/health_generate")
            assert reply.status_code == 503
            assert reply.json()["probe_ok"] is True
            assert reply.json()["stalled"] is True
            assert reply.json()["ok"] is False
            assert (await client.get("/ready")).status_code == 503
            assert (await client.post("/health_generate")).status_code == 405
            tm.rid_to_state.clear()
            assert (await client.get("/health_generate")).status_code == 200
            assert (await client.get("/ready")).status_code == 200


class TestRealCoreProbe:
    @pytest.mark.parametrize("queue_name", ["waiting", "bootstrap"])
    @pytest.mark.parametrize("window", [0, WINDOW])
    async def test_completed_prefill_chunks_keep_probes_healthy_until_work_stops(
        self, tm, clock, native_probe, completed_work, monkeypatch, queue_name, window
    ):
        # Exercise the actual scheduler helper and core probe.
        # The long request produces no token while its chunks complete. Only
        # IPC delivery is simulated, at handle_loop's receive-timestamp boundary.
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.waiting_queue = []
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
        queue = (
            scheduler.waiting_queue
            if queue_name == "waiting"
            else scheduler.disagg_prefill_bootstrap_queue.queue
        )
        chunks = SimpleNamespace(complete=True, seen=set(), signals=[])

        def receive(output):
            assert isinstance(output, HealthCheckOutput)
            tm._result_dispatcher(output)
            if output.rid != hg.COMPLETED_WORK_RID:
                chunks.signals.append(output.rid)
            tm.last_receive_tstamp = clock.time()

        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=receive)
        )
        completed_batch = SimpleNamespace(
            reqs=[SimpleNamespace(rid="long-prompt", http_worker_ipc=None)],
            forward_mode=ForwardMode.EXTEND,
        )
        advance = clock.advance

        def tick(seconds):
            advance(seconds)
            for probe in native_probe.probes:
                if probe.rid not in chunks.seen:
                    queue.append(SimpleNamespace(rid=probe.rid, http_worker_ipc=None))
                    chunks.seen.add(probe.rid)
            if chunks.complete:
                scheduler.maybe_signal_queued_health_checks(completed_batch)

        monkeypatch.setattr(clock, "advance", tick)
        native_probe.acknowledge = False
        req = state(clock, "long-prompt")
        tm.rid_to_state[req.obj.rid] = req

        with WINDOW_ENV.override(window):
            # The chain outlasts the core timeout, but every probe sees a newly
            # completed chunk. A first token is not needed to remain healthy.
            polls = int(window) + 2 if window else http_server.HEALTH_CHECK_TIMEOUT + 1
            for _ in range(polls):
                assert (await native_probe.poll()).status_code == 200
                assert tm.is_ready()
            assert req.time_stats.first_token_time == 0
            if window:
                assert tm._tpu_pd_last_progress_time == clock.perf_counter()
            else:
                assert not hasattr(tm, "_tpu_pd_last_progress_time")
            assert len(chunks.signals) == len(native_probe.probes)
            assert len(set(chunks.signals)) == len(chunks.signals)

            # A prior chunk acknowledgment cannot keep the next probe alive.
            chunks.complete = False
            sent = len(chunks.signals)
            response = await native_probe.poll()
            if window:
                assert response.status_code == 200
                assert msgspec.json.decode(response.body)["probe_ok"] is False
                clock.advance(window)
                response = await native_probe.poll()
            assert response.status_code == 503
            assert msgspec.json.decode(response.body)["probe_ok"] is False
            assert len(chunks.signals) == sent
            assert (await http_server.ready()).status_code == 503

            chunks.complete = True
            assert (await native_probe.poll()).status_code == 200
            assert (await http_server.ready()).status_code == 200
        assert set(tm.rid_to_state) == {req.obj.rid}
        assert all(task.done() for task in native_probe.tasks)

    @pytest.mark.parametrize("window", [0, WINDOW])
    async def test_real_probe_timeout_and_recovery(
        self, tm, clock, native_probe, window
    ):
        native_probe.acknowledge = False
        before = clock.perf_counter()
        with WINDOW_ENV.override(window):
            response = await native_probe.poll()
        assert response.status_code == 503
        assert clock.perf_counter() - before == 3
        assert not tm.rid_to_state
        assert not tm.is_ready()
        assert (await http_server.ready()).status_code == 503
        assert msgspec.json.decode(response.body)["probe_ok"] is False
        assert msgspec.json.decode(response.body)["ok"] is False
        native_probe.acknowledge = True
        with WINDOW_ENV.override(window):
            assert (await native_probe.poll()).status_code == 200
        assert tm.is_ready()
        assert (await http_server.ready()).status_code == 200


class TestCompletedWorkWindow:
    @pytest.mark.parametrize("chunk_first", [False, True])
    async def test_chunks_and_tokens_refresh_the_same_progress_window(
        self, tm, clock, route, completed_work, chunk_first
    ):
        tm.rid_to_state["real"] = state(clock, "real", age=1000)
        for is_chunk in (chunk_first, not chunk_first):
            if is_chunk:
                completed_work()
            else:
                await tm._handle_batch_output(batch("real"))
            clock.advance(WINDOW - 10)
        assert (await hg.health_generate(request())).status_code == 200
        clock.advance(11)
        tm._result_dispatcher(HealthCheckOutput())
        response = await hg.health_generate(request())
        assert response.status_code == 503
        assert msgspec.json.decode(response.body)["stalled"] is True

    @pytest.mark.parametrize("probe_ok", [False, True])
    @pytest.mark.parametrize("age", [99.999, 100, 100.001])
    @pytest.mark.parametrize("source", ["chunk", "buffered_token", "output"])
    async def test_real_progress_window_boundaries(
        self, tm, clock, route, completed_work, probe_ok, age, source
    ):
        tm.rid_to_state["real"] = state(clock, "real", age=200)
        if source == "output":
            await tm._handle_batch_output(batch("real"))
        else:
            completed_work(
                ForwardMode.EXTEND if source == "chunk" else ForwardMode.DECODE
            )
        clock.advance(age)
        route.side_effect = None
        route.return_value = Response(status_code=200 if probe_ok else 503)
        response = await hg.health_generate(request())
        assert response.status_code == (200 if age <= WINDOW else 503)
        assert tm.is_ready() is (age <= WINDOW)
        health = hg.verdict(tm, probe_ok=probe_ok, window=WINDOW)
        assert health.idle_s == age
        assert health.stalled is (age > WINDOW)

    @pytest.mark.parametrize("probe_ok", [False, True])
    @pytest.mark.parametrize("age", [0, 99.999, 100, 100.001])
    async def test_first_dispatch_gets_one_full_window(
        self, tm, clock, route, probe_ok, age
    ):
        tm.rid_to_state["new"] = state(clock, age=age)
        route.side_effect = None
        route.return_value = Response(status_code=200 if probe_ok else 503)
        response = await hg.health_generate(request())
        assert response.status_code == (200 if age <= WINDOW else 503)

    async def test_synthetic_probes_cannot_hide_stuck_real_work(
        self, tm, clock, native_probe
    ):
        tm.rid_to_state["waiting-for-kv"] = state(clock, age=WINDOW + 1)
        for _ in range(3):
            response = await native_probe.poll()
            payload = msgspec.json.decode(response.body)
            assert payload["probe_ok"] is True
            assert payload["stalled"] is True
            assert response.status_code == 503
        assert not hasattr(tm, "_tpu_pd_last_progress_time")
        assert (await http_server.ready()).status_code == 503

    async def test_new_arrivals_do_not_reset_an_existing_stall(self, tm, clock, route):
        tm.rid_to_state["old"] = state(clock, age=WINDOW + 1)
        for i in range(3):
            tm.rid_to_state[str(i)] = state(clock)
            assert (await hg.health_generate(request())).status_code == 503

    async def test_abort_outputs_do_not_keep_pending_work_healthy(
        self, tm, clock, route, wrapped
    ):
        tm.rid_to_state["waiting"] = state(clock, age=WINDOW + 1)
        for i in range(3):
            rid = f"aborted-{i}"
            tm.rid_to_state[rid] = state(clock, rid)
            await tm._handle_batch_output(
                batch(rid, finished={"type": "abort", "message": "timeout"})
            )
            assert (await hg.health_generate(request())).status_code == 503
        assert not hasattr(tm, "_tpu_pd_last_progress_time")

    @pytest.mark.parametrize("probe_ok", [False, True])
    async def test_hung_scheduler_does_not_acknowledge_abort(
        self, tm, clock, route, monkeypatch, probe_ok
    ):
        pending = state(clock, "pending")
        tm.rid_to_state["pending"] = pending
        send = Mock()
        monkeypatch.setattr(tm, "_dispatch_to_scheduler", send)
        tm.abort_request("pending")
        tm.abort_request("pending")
        send.assert_called_once()
        assert isinstance(send.call_args.args[0], AbortReq)
        assert pending.abort_sent and not pending.finished
        clock.advance(WINDOW + 1)
        route.side_effect = None
        route.return_value = Response(status_code=200 if probe_ok else 503)
        response = await hg.health_generate(request())
        assert response.status_code == 503
        assert msgspec.json.decode(response.body)["stalled"] is True
        assert not tm.is_ready()

        tm._handle_abort_req(AbortReq(rid="pending"))
        route.return_value = Response(status_code=200)
        assert (await hg.health_generate(request())).status_code == 200
        assert not tm.rid_to_state

    @pytest.mark.parametrize("register_after_init", [False, True])
    def test_progress_dispatcher_preserves_worker_handler_registration(
        self, tm, clock, wrapped, monkeypatch, register_after_init
    ):
        handle = Mock(return_value=object())

        def register():
            tm._result_dispatcher += TypeBasedDispatcher(
                [(PauseContinueBroadcastReq, handle)]
            )

        monkeypatch.setattr(
            tm, "init_communicators", Mock() if register_after_init else register
        )
        tm.init_request_dispatcher()
        hg.install_health_generate_route.assert_called_once_with()
        if register_after_init:
            register()
        message = PauseContinueBroadcastReq(is_pause=True)
        assert tm._result_dispatcher(message) is handle.return_value
        handle.assert_called_once_with(message)
        assert not hasattr(tm, "_tpu_pd_last_progress_time")
        tm._result_dispatcher(HealthCheckOutput(rid=hg.COMPLETED_WORK_RID))
        assert tm._tpu_pd_last_progress_time == clock.perf_counter()

    @pytest.mark.parametrize("rid", [None, "HEALTH_CHECK_x", hg.COMPLETED_WORK_RID])
    def test_only_completed_work_survives_ipc_as_progress(self, tm, clock, rid):
        output = HealthCheckOutput(rid=rid, http_worker_ipc="ipc:///worker")
        wire = msgpack_decode(msgpack_encode(msgpack_decode(msgpack_encode(output))))
        assert wire == output
        hg.mark_progress(tm, wire)
        expected = clock.perf_counter() if rid == hg.COMPLETED_WORK_RID else None
        assert getattr(tm, "_tpu_pd_last_progress_time", None) == expected

    @pytest.mark.parametrize(
        "kind", ["empty", "probe", "aborted", "pending_abort", "finished", "mixed"]
    )
    def test_progress_filters_probes_and_aborts_and_preserves_worker_routes(self, kind):
        send = Mock()
        scheduler = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.PREFILL,
            ipc_channels=SimpleNamespace(
                send_to_tokenizer=SimpleNamespace(send_output=send)
            ),
        )
        real = SimpleNamespace(
            rid="real", http_worker_ipc="ipc:///a", finished_reason=None
        )
        probe = SimpleNamespace(
            rid=HEALTH_CHECK_RID_PREFIX + "_x", http_worker_ipc="ipc:///probe"
        )
        reqs = [real]
        if kind == "empty":
            reqs = []
        elif kind == "probe":
            reqs = [probe]
        elif kind == "aborted":
            real.finished_reason = FINISH_ABORT()
        elif kind == "pending_abort":
            real.to_finish = FINISH_ABORT()
        elif kind == "finished":
            real.finished_reason = FINISH_LENGTH(1)
        else:
            reqs += [
                real,
                probe,
                SimpleNamespace(rid="other", http_worker_ipc="ipc:///b"),
            ]
        hg.report_completed_work(
            scheduler, SimpleNamespace(forward_mode=ForwardMode.EXTEND, reqs=reqs)
        )
        expected = (
            {"ipc:///a", "ipc:///b"}
            if kind == "mixed"
            else {"ipc:///a"}
            if kind == "finished"
            else set()
        )
        assert {
            call.args[0].http_worker_ipc for call in send.call_args_list
        } == expected
        assert send.call_count == len(expected)

    @pytest.mark.parametrize("fail", [False, True])
    def test_progress_is_sent_after_batch_completion_and_before_probe_ack(
        self, tm, completed_work, monkeypatch, fail
    ):
        events = []
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.scheduler_stage_metrics = None
        scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        scheduler.waiting_queue = []
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
        scheduler.publish_load_snapshot = Mock()
        scheduler.load_publisher = Mock()
        scheduler.load_inquirer = Mock()
        scheduler.tree_cache = Mock()
        scheduler._record_step_counters = Mock()
        scheduler.metrics_reporter = Mock()
        scheduler.enable_fpm = False
        scheduler._maybe_clear_mm_inputs = Mock()
        scheduler.kv_free_fence = None
        scheduler.return_health_check_ipcs = deque(["ipc:///probe"])
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(
                send_output=lambda obj: events.append(
                    "progress" if obj.rid == hg.COMPLETED_WORK_RID else "ack"
                )
            )
        )

        def finish(batch, result):
            if fail:
                raise RuntimeError("compute failed")
            events.append("completed")

        scheduler.process_batch_result_disagg_prefill = finish
        monkeypatch.setattr(scheduler_module, "flush_trace_batch", Mock())
        completed = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            is_dllm=lambda: False,
            reqs=[SimpleNamespace(rid="real", http_worker_ipc=None)],
        )
        if fail:
            with pytest.raises(RuntimeError, match="compute failed"):
                scheduler.process_batch_result(completed, object())
            assert events == []
        else:
            scheduler.process_batch_result(completed, object())
            assert events == ["completed", "progress", "ack"]


class TestConfirmedWorkerFailure:
    async def test_crash_during_probe_is_detected(self, tm, route):
        worker = SimpleNamespace(exitcode=None)
        tm._subprocess_watchdog = SubprocessWatchdog([worker], ["scheduler"])

        async def probe(request):
            worker.exitcode = -9
            return Response(status_code=200)

        route.side_effect = probe
        response = await hg.health_generate(request())
        assert response.status_code == 503
        assert msgspec.json.decode(response.body)["worker_failures"] == [
            "scheduler (exitcode -9)"
        ]
        route.assert_awaited_once()

    @pytest.mark.parametrize("exitcode", [1, -9, 42])
    async def test_confirmed_crash_bypasses_a_hung_probe_despite_recent_work(
        self, tm, clock, route, completed_work, exitcode
    ):
        tm.rid_to_state["live"] = state(clock, age=1000)
        completed_work()
        tm._subprocess_watchdog = SubprocessWatchdog(
            [SimpleNamespace(exitcode=exitcode)], ["scheduler"]
        )

        async def blocked_probe(request):
            await asyncio.Event().wait()

        route.side_effect = blocked_probe
        response = await asyncio.wait_for(hg.health_generate(request()), timeout=1)
        assert response.status_code == 503
        payload = msgspec.json.decode(response.body)
        assert payload["probe_ok"] is None
        assert payload["stalled"] is False
        assert payload["worker_failures"] == [f"scheduler (exitcode {exitcode})"]
        assert not tm.is_ready()
        route.assert_not_awaited()


class TestNoFalseUnhealthy:
    @pytest.mark.parametrize("kind", ["text", "tokens"])
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        "metrics,log_metrics",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    async def test_progress_is_independent_of_metrics_flags(
        self, tm, clock, route, wrapped, kind, stream, metrics, log_metrics
    ):
        tm.enable_metrics = metrics
        req = state(clock, "live", log_metrics=log_metrics)
        req.obj.stream = stream
        tm.rid_to_state.update(live=req, queued=state(clock, "queued"))
        await tm._handle_batch_output(batch("live", tokens=1, kind=kind))
        # Production continues every second, far more often than the window.
        for tokens in range(2, 103):
            clock.advance(1)
            await tm._handle_batch_output(batch("live", tokens=tokens, kind=kind))
        assert len(req.output_ids) == 102
        assert not hg.verdict(tm, probe_ok=True, window=WINDOW).stalled
        response = await hg.health_generate(request())
        assert response.status_code == 200, (
            f"Continuous output must remain healthy with enable_metrics={metrics}, log_metrics={log_metrics}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


class TestBoundaryCases:
    """Edges around the window, the clock and concurrent probes."""

    async def test_hang_after_first_token_is_stalled_once_progress_is_a_window_old(
        self, tm, clock, route
    ):
        # Busy hang: tokens flowed, then nothing. The table alone must decide
        # at the window, and the next output must clear it at once.
        tm.rid_to_state["r"] = state(clock, "r", age=10, produced=5)
        tm._tpu_pd_last_progress_time = clock.perf_counter() - 5
        assert (await hg.health_generate(request())).status_code == 200
        clock.advance(WINDOW)
        response = await hg.health_generate(request())
        assert response.status_code == 503
        assert "stalled" in msgspec.json.decode(response.body)["reason"]
        assert not tm.is_ready()
        hg.mark_progress(tm, batch("r"))
        assert (await hg.health_generate(request())).status_code == 200
        assert tm.is_ready()

    async def test_wall_clock_jumps_do_not_change_the_verdict(self, tm, clock, route):
        # Dispatch and progress are perf_counter based; only real_time moves here.
        tm.rid_to_state["r"] = state(clock, "r", age=WINDOW / 2)
        for jump in (-86400, 86400):
            clock.wall += jump
            tm.last_receive_tstamp = clock.time()
            assert (await hg.health_generate(request())).status_code == 200
            assert not hg.verdict(tm, probe_ok=True, window=WINDOW).stalled

    async def test_concurrent_probes_agree_and_leave_the_server_up(
        self, tm, clock, route
    ):
        tm.rid_to_state["r"] = state(clock, "r", age=1, produced=0.5)
        responses = await asyncio.gather(
            *(hg.health_generate(request()) for _ in range(5))
        )
        assert [r.status_code for r in responses] == [200] * 5
        assert route.await_count == 5
        assert tm.server_status == ServerStatus.Up

    @pytest.mark.parametrize("window", [0, WINDOW])
    async def test_probe_timeout_while_busy_depends_on_the_window(
        self, tm, clock, route, window
    ):
        # With a window, active work is judged by progress and a slow probe is
        # tolerated; with the table check off, the probe is the only judge.
        tm.rid_to_state["r"] = state(clock, "r", age=5)
        route.side_effect = None
        route.return_value = Response(status_code=503)
        with WINDOW_ENV.override(window):
            response = await hg.health_generate(request())
        assert response.status_code == (200 if window else 503)
