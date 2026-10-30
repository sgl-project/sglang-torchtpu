"""The mini load balancer must return an upstream rejection with its own
status instead of wrapping it in a 200 event stream, and must close its
upstream session however a request ends. A streaming prefill that fails in
its 200 body is returned as is, without waiting for decode (#1706)."""

import asyncio
import contextlib
import json
import socket
import time
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from sglang_router import mini_lb
from sglang_router.mini_lb import MiniLoadBalancer
from sglang_router.router_args import RouterArgs


async def _serve(handler):
    app = web.Application()
    app.router.add_post("/generate", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


async def _reject(request):
    return web.json_response(
        {"error": {"message": "The input is longer than the context length"}},
        status=400,
    )


async def _stream(request):
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    for i in range(3):
        await resp.write(json.dumps({"text": "x" * (i + 1)}).encode() + b"\n\n")
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


def _lb(prefill_url, decode_url):
    args = SimpleNamespace(
        host="127.0.0.1",
        port=0,
        policy="random",
        mini_lb_policy="random",
        pd_disaggregation=True,
        prefill_urls=[(prefill_url, 8998)],
        decode_urls=[decode_url],
        test_external_dp_routing=False,
        request_timeout_secs=10,
        disable_health_check=True,
        health_check_interval_secs=60,
    )
    return MiniLoadBalancer(args)


async def _run(
    prefill_handler, decode_handler, max_chunks=None, request=None, before_cleanup=None
):
    """Issue one streaming request (default `{"text": "hello", "stream": True}`)
    through MiniLB and read up to `max_chunks` of its body; returns the
    response, the chunks read and the upstream sessions MiniLB opened.
    `before_cleanup` is awaited before the test servers shut down."""
    sessions = []
    real_session = aiohttp.ClientSession

    def recording_session(*args, **kwargs):
        sessions.append(real_session(*args, **kwargs))
        return sessions[-1]

    prefill_runner, prefill_url = await _serve(prefill_handler)
    decode_runner, decode_url = await _serve(decode_handler)
    aiohttp.ClientSession = recording_session
    try:
        lb = _lb(prefill_url, decode_url)
        response = await lb.generate_stream(
            request or {"text": "hello", "stream": True}, prefill_url, decode_url
        )
        chunks = []
        if hasattr(response, "body_iterator"):
            async for chunk in response.body_iterator:
                chunks.append(chunk)
                if max_chunks is not None and len(chunks) >= max_chunks:
                    await response.body_iterator.aclose()
                    break
        if before_cleanup is not None:
            await before_cleanup()
        return response, chunks, sessions
    finally:
        aiohttp.ClientSession = real_session
        await prefill_runner.cleanup()
        await decode_runner.cleanup()


def _assert_sessions_closed(sessions):
    assert len(sessions) == 1
    assert sessions[0].closed


@pytest.mark.unit
def test_rejected_request_keeps_upstream_status():
    response, chunks, sessions = asyncio.run(_run(_reject, _reject))
    assert response.status_code == 400
    assert b"longer than the context length" in response.body
    assert chunks == []
    _assert_sessions_closed(sessions)


@pytest.mark.unit
def test_prefill_rejection_is_reported_even_if_decode_streams():
    response, chunks, sessions = asyncio.run(_run(_reject, _stream))
    assert response.status_code == 400
    assert chunks == []
    _assert_sessions_closed(sessions)


@pytest.mark.unit
def test_accepted_request_streams_decode_output():
    response, chunks, sessions = asyncio.run(_run(_stream, _stream))
    assert response.status_code == 200
    body = b"".join(chunks)
    assert body.count(b'"text"') == 3
    assert body.endswith(b"data: [DONE]\n\n")
    _assert_sessions_closed(sessions)


@pytest.mark.unit
def test_abandoned_stream_closes_session():
    response, chunks, sessions = asyncio.run(_run(_stream, _stream, max_chunks=1))
    assert response.status_code == 200
    assert len(chunks) == 1
    _assert_sessions_closed(sessions)


@contextlib.asynccontextmanager
async def _engine(handler=None, info_handler=None):
    async def healthy(request):
        payload = await request.json()
        if payload.get("stream"):
            return web.Response(
                body=b'data: {"text":"ok"}\n\ndata: [DONE]\n\n',
                content_type="text/event-stream",
            )
        response = {"text": "ok"}
        if "routed_dp_rank" in payload:
            response["meta_info"] = {"dp_rank": payload["routed_dp_rank"]}
        return web.json_response(response)

    async def info(_request):
        return web.json_response({"internal_states": [{}]})

    async def loads(_request):
        return web.json_response(
            {
                "loads": [
                    {
                        "num_running_reqs": 0,
                        "num_waiting_reqs": 0,
                        "num_waiting_uncached_tokens": 0,
                    }
                ]
            }
        )

    app = web.Application()
    app.router.add_post("/generate", handler or healthy)
    app.router.add_get("/server_info", info_handler or info)
    app.router.add_get("/v1/loads", loads)
    async with TestServer(app) as server:
        yield str(server.make_url("")).rstrip("/")


def _affinity_lb(prefill, decode):
    return MiniLoadBalancer(
        RouterArgs(
            mini_lb_policy="session_affinity",
            pd_disaggregation=True,
            prefill_urls=[(prefill, 8998)],
            decode_urls=[decode],
            request_timeout_secs=1,
        )
    )


@pytest.mark.unit
@pytest.mark.parametrize("failing_role", ["prefill", "decode"])
@pytest.mark.parametrize(
    ("stream", "external_dp"), [(False, False), (True, False), (False, True)]
)
def test_affinity_connect_timeout_bounds_engine_requests(
    monkeypatch, failing_role, stream, external_dp
):
    """A silent TCP connect must stop before the longer request deadline."""
    real_connect = aiohttp.TCPConnector._create_connection
    sessions = []
    real_session = aiohttp.ClientSession

    async def stalled_connect(connector, request, traces, timeout):
        if request.url.host == "never-connect.invalid":
            await asyncio.Event().wait()
        return await real_connect(connector, request, traces, timeout)

    def record_session(*args, **kwargs):
        session = real_session(*args, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(aiohttp.TCPConnector, "_create_connection", stalled_connect)
    monkeypatch.setattr(aiohttp, "ClientSession", record_session)
    assert mini_lb._CONNECT_TIMEOUT_SECS == 10
    monkeypatch.setattr(mini_lb, "_CONNECT_TIMEOUT_SECS", 0.02)

    async def run():
        async with _engine() as healthy:
            dead = "http://never-connect.invalid:30000"
            prefill, decode = (
                (dead, healthy) if failing_role == "prefill" else (healthy, dead)
            )
            balancer = _affinity_lb(prefill, decode)
            balancer.test_external_dp_routing = external_dp
            forward = balancer.generate_stream if stream else balancer.generate
            with pytest.raises(aiohttp.ClientConnectionError):
                await asyncio.wait_for(
                    forward(
                        {"text": "hi", "stream": stream}, prefill, decode, "generate"
                    ),
                    timeout=0.5,
                )
            assert sessions and all(session.closed for session in sessions)
            assert all(
                session.timeout.connect == session.timeout.sock_connect == 0.02
                for session in sessions
            )
            assert all(session.timeout.total == 1 for session in sessions)

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize("failing_role", ["prefill", "decode"])
@pytest.mark.parametrize("stream", [False, True])
def test_affinity_refused_engine_returns_502(monkeypatch, failing_role, stream):
    """A refused connection is attributed to its engine and returned as 502."""

    async def run():
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            dead = f"http://127.0.0.1:{reserved.getsockname()[1]}"
            async with _engine() as healthy:
                prefill, decode = (
                    (dead, healthy) if failing_role == "prefill" else (healthy, dead)
                )
                balancer = _affinity_lb(prefill, decode)
                monkeypatch.setattr(mini_lb, "lb", balancer)
                response = await mini_lb._forward_to_backend(
                    {"text": "hi", "stream": stream}, "generate", "s-1"
                )
                assert response.status_code == 502
                assert json.loads(response.body)["error"]["message"].startswith(
                    f"upstream unreachable: {dead}:"
                )
                assert not balancer.affinity_policy.loads.is_reachable(dead)
                assert balancer.affinity_policy.loads.is_reachable(healthy)

    asyncio.run(run())


@pytest.mark.unit
def test_affinity_failed_dp_metadata_does_not_cache_partial_sizes(monkeypatch):
    """A failed metadata query must not leave a half-initialized DP cache."""

    async def run():
        available = False

        async def metadata(request):
            if not available:
                request.transport.close()
                return web.Response()
            return web.json_response({"internal_states": [{}]})

        async with _engine() as prefill, _engine(info_handler=metadata) as decode:
            balancer = _affinity_lb(prefill, decode)
            balancer.test_external_dp_routing = True
            monkeypatch.setattr(mini_lb, "lb", balancer)
            failed = await mini_lb._forward_to_backend({"text": "hi"}, "generate", None)
            assert failed.status_code == 502
            assert balancer.prefill_dp_size is balancer.decode_dp_size is None
            assert not balancer.affinity_policy.loads.is_reachable(decode)
            available = True
            async with aiohttp.ClientSession() as session:
                await balancer._load_poller._poll(session=session, url=decode)
            recovered = await mini_lb._forward_to_backend(
                {"text": "hi"}, "generate", None
            )
            assert recovered.status_code == 200
            assert balancer.prefill_dp_size == balancer.decode_dp_size == 1

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize("stream", [False, True])
def test_affinity_disconnect_cancels_peer_without_retry(monkeypatch, stream):
    """An early disconnect must not leave its peer request waiting for headers."""

    async def run():
        peer_started = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def peer(request):
            calls.append("peer")
            peer_started.set()
            await release.wait()
            return web.json_response({"text": "late"})

        async def failing(request):
            calls.append("failing")
            await peer_started.wait()
            request.transport.close()
            return web.Response()

        async with _engine(peer) as prefill, _engine(failing) as decode:
            balancer = _affinity_lb(prefill, decode)
            monkeypatch.setattr(mini_lb, "lb", balancer)
            try:
                response = await asyncio.wait_for(
                    mini_lb._forward_to_backend({"stream": stream}, "generate", None),
                    timeout=0.5,
                )
                assert response.status_code == 502
                assert sorted(calls) == ["failing", "peer"]
                assert balancer.affinity_policy.loads.is_reachable(prefill)
                assert not balancer.affinity_policy.loads.is_reachable(decode)
                assert not [
                    task
                    for task in asyncio.all_tasks()
                    if task.get_coro().__qualname__
                    == "MiniLoadBalancer._engine_request"
                ]
            finally:
                release.set()

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize("failing_role", ["prefill", "decode"])
def test_affinity_truncated_body_returns_502(monkeypatch, failing_role):
    """Losing a JSON response body before client output is still a connection failure."""

    async def truncated(request):
        response = web.StreamResponse(
            headers={"Content-Type": "application/json", "Content-Length": "100"}
        )
        await response.prepare(request)
        await response.write(b'{"text":')
        request.transport.close()
        return response

    async def run():
        async with _engine() as healthy, _engine(truncated) as failing:
            prefill, decode = (
                (failing, healthy) if failing_role == "prefill" else (healthy, failing)
            )
            balancer = _affinity_lb(prefill, decode)
            monkeypatch.setattr(mini_lb, "lb", balancer)
            response = await mini_lb._forward_to_backend(
                {"text": "hi", "return_logprob": True}, "generate", None
            )
            assert response.status_code == 502
            assert not balancer.affinity_policy.loads.is_reachable(failing)
            assert balancer.affinity_policy.loads.is_reachable(healthy)

    asyncio.run(run())


@pytest.mark.unit
def test_affinity_request_deadline_does_not_mark_engine_unreachable():
    """A total request deadline does not establish a connection failure."""

    async def run():
        release = asyncio.Event()

        async def slow(_request):
            await release.wait()
            return web.json_response({"text": "late"})

        async with _engine() as prefill, _engine(slow) as decode:
            balancer = _affinity_lb(prefill, decode)
            balancer.timeout = 0.03
            try:
                with pytest.raises(asyncio.TimeoutError):
                    await balancer.generate({"text": "hi"}, prefill, decode, "generate")
                assert balancer.affinity_policy.loads.is_reachable(prefill)
                assert balancer.affinity_policy.loads.is_reachable(decode)
            finally:
                release.set()

    asyncio.run(run())


# --- #658: a side that fails first must not leave the other side working ---

SLOW_SECONDS = 5.0
EARLY_RETURN_BOUND_SECONDS = 2.0


def _slow(disconnected: asyncio.Event):
    """A server that takes SLOW_SECONDS to answer (like a prefill computing a
    long prompt) and sets `disconnected` once MiniLB closes the connection."""

    async def handler(request):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + SLOW_SECONDS
        try:
            while loop.time() < deadline:
                if request.transport is None or request.transport.is_closing():
                    disconnected.set()
                    break
                await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            # Some aiohttp versions cancel the handler on client disconnect.
            disconnected.set()
            raise
        return web.json_response({"text": "late"})

    return handler


async def _run_early_failure(slow_side, stream):
    """Send one request whose `slow_side` ("prefill" or "decode") is slow and
    whose other side rejects immediately; return (response, elapsed_s,
    slow_side_disconnected)."""
    disconnected = asyncio.Event()
    slow = _slow(disconnected)
    prefill_handler, decode_handler = (
        (slow, _reject) if slow_side == "prefill" else (_reject, slow)
    )
    prefill_runner, prefill_url = await _serve(prefill_handler)
    decode_runner, decode_url = await _serve(decode_handler)
    try:
        lb = _lb(prefill_url, decode_url)
        start = time.monotonic()
        if stream:
            response = await lb.generate_stream(
                {"text": "hello", "stream": True}, prefill_url, decode_url
            )
        else:
            response = await lb.generate(
                {"text": "hello"}, prefill_url, decode_url, "generate"
            )
        elapsed = time.monotonic() - start
        try:
            await asyncio.wait_for(disconnected.wait(), EARLY_RETURN_BOUND_SECONDS)
        except asyncio.TimeoutError:
            pass
        return response, elapsed, disconnected.is_set()
    finally:
        await prefill_runner.cleanup()
        await decode_runner.cleanup()


@pytest.mark.unit
@pytest.mark.parametrize("stream", [False, True])
def test_early_decode_rejection_cancels_slow_prefill(stream):
    response, elapsed, disconnected = asyncio.run(_run_early_failure("prefill", stream))
    assert response.status_code == 400
    assert b"longer than the context length" in response.body
    assert elapsed < EARLY_RETURN_BOUND_SECONDS
    assert disconnected, "MiniLB must close its connection to the doomed prefill"


@pytest.mark.unit
@pytest.mark.parametrize("stream", [False, True])
def test_early_prefill_rejection_cancels_slow_decode(stream):
    response, elapsed, disconnected = asyncio.run(_run_early_failure("decode", stream))
    assert response.status_code == 400
    assert elapsed < EARLY_RETURN_BOUND_SECONDS
    assert disconnected, "MiniLB must close its connection to the doomed decode"


@pytest.mark.unit
def test_generate_returns_decode_rejection_after_fast_prefill():
    """When prefill finishes 200 first and decode then rejects with non-200,
    both responses are non-None; generate() must still return decode's status."""

    async def fast_prefill(request):
        return web.json_response({"text": "", "meta_info": {}})

    async def slow_reject_decode(request):
        await asyncio.sleep(0.05)
        return web.Response(text="decode overloaded", status=503)

    async def run():
        prefill_runner, prefill_url = await _serve(fast_prefill)
        decode_runner, decode_url = await _serve(slow_reject_decode)
        try:
            lb = _lb(prefill_url, decode_url)
            return await lb.generate(
                {"text": "hello"}, prefill_url, decode_url, "generate"
            )
        finally:
            await prefill_runner.cleanup()
            await decode_runner.cleanup()

    response = asyncio.run(run())
    assert response.status_code == 503
    assert response.body == b"decode overloaded"


# --- #1706: a streaming prefill reports its failure in a 200 body ---

DECODE_KV_WAIT_SECONDS = 3.0
PREFILL_FAILURE_BOUND_SECONDS = 1.5


def _sse(*events):
    """Encode events as an SSE body ending in `[DONE]`, like the engine."""
    return b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) + (
        b"data: [DONE]\n\n"
    )


PREFILL_ERROR_BODY = _sse(
    {
        "error": {
            "message": "The input is longer than the context length",
            "type": "invalid_request_error",
            "code": 400,
            "retryable": False,
        }
    }
)
PREFILL_ABORT_BODY = _sse(
    {
        "text": "",
        "meta_info": {
            "id": "r1",
            "finish_reason": {
                "type": "abort",
                "message": "Request r1 exceeds the maximum number of tokens: 9 > 8",
                "status_code": 400,
            },
        },
    }
)
PREFILL_FINISHED_BODY = _sse(
    {
        "text": "",
        "meta_info": {
            "id": "r1",
            "finish_reason": {"type": "length", "length": 1},
            "input_token_logprobs": [[-0.5, 1, None], [-0.25, 2, None]],
        },
    }
)
DECODE_BODY = _sse(
    *(
        {
            "text": "x" * (i + 1),
            "meta_info": {"id": "r1", "input_token_logprobs": []},
        }
        for i in range(3)
    )
)


def _prefill_answers(body):
    """A streaming prefill that answers 200 with `body`."""

    async def handler(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(body)
        await resp.write_eof()
        return resp

    return handler


def _decode_waiting_for_kv(disconnected: asyncio.Event, wait_seconds):
    """A decode that answers its headers at once, then streams DECODE_BODY
    after `wait_seconds` (waiting for prefill's KV); sets `disconnected` if
    MiniLB closes the connection first."""

    async def handler(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_seconds
        try:
            while loop.time() < deadline:
                if request.transport is None or request.transport.is_closing():
                    disconnected.set()
                    return resp
                await asyncio.sleep(0.02)
            await resp.write(DECODE_BODY)
            await resp.write_eof()
        except asyncio.CancelledError:
            disconnected.set()
            raise
        except ConnectionError:
            disconnected.set()
        return resp

    return handler


async def _run_with_prefill_body(prefill_body, decode_wait_seconds, request=None):
    """Stream one request whose prefill answers 200 with `prefill_body`; return
    (response, body, elapsed_s including the body, decode_disconnected,
    sessions). Disconnection is sampled before the test servers shut down,
    since their shutdown would disconnect decode anyway."""
    disconnected = asyncio.Event()
    observed = {}

    async def before_cleanup():
        observed["elapsed"] = time.monotonic() - start
        try:
            await asyncio.wait_for(disconnected.wait(), 1.0)
        except asyncio.TimeoutError:
            pass
        observed["disconnected"] = disconnected.is_set()

    start = time.monotonic()
    response, chunks, sessions = await _run(
        _prefill_answers(prefill_body),
        _decode_waiting_for_kv(disconnected, decode_wait_seconds),
        request=request,
        before_cleanup=before_cleanup,
    )
    # A failed prefill is answered with a plain (non-streaming) Response.
    body = b"".join(chunks) if hasattr(response, "body_iterator") else response.body
    return response, body, observed["elapsed"], observed["disconnected"], sessions


def _assert_prefill_failure_returned(prefill_body, request=None):
    response, body, elapsed, decode_disconnected, sessions = asyncio.run(
        _run_with_prefill_body(prefill_body, DECODE_KV_WAIT_SECONDS, request)
    )
    assert response.status_code == 200
    assert response.media_type == "text/event-stream"
    assert body == prefill_body
    assert elapsed < PREFILL_FAILURE_BOUND_SECONDS, "MiniLB waited for decode"
    assert decode_disconnected, "MiniLB must close its connection to the decode"
    _assert_sessions_closed(sessions)


@pytest.mark.unit
def test_prefill_inband_error_does_not_wait_for_decode():
    _assert_prefill_failure_returned(PREFILL_ERROR_BODY)


@pytest.mark.unit
def test_aborted_prefill_does_not_wait_for_decode():
    _assert_prefill_failure_returned(PREFILL_ABORT_BODY)


@pytest.mark.unit
def test_failed_prefill_with_return_logprob_returns_prefill_stream():
    _assert_prefill_failure_returned(
        PREFILL_ERROR_BODY,
        request={"text": "hello", "stream": True, "return_logprob": True},
    )


@pytest.mark.unit
def test_finished_prefill_still_relays_decode():
    response, body, _, decode_disconnected, sessions = asyncio.run(
        _run_with_prefill_body(PREFILL_FINISHED_BODY, 0.3)
    )
    assert response.status_code == 200
    assert body == DECODE_BODY
    assert not decode_disconnected
    _assert_sessions_closed(sessions)


@pytest.mark.unit
def test_finished_prefill_with_return_logprob_merges_input_logprobs():
    response, body, _, _, sessions = asyncio.run(
        _run_with_prefill_body(
            PREFILL_FINISHED_BODY,
            0.3,
            request={"text": "hello", "stream": True, "return_logprob": True},
        )
    )
    assert response.status_code == 200
    # Parse per line: the merge path may re-emit blank separator lines.
    events = [
        line[len(b"data:") :].strip()
        for line in body.split(b"\n")
        if line.startswith(b"data:")
    ]
    assert events[-1] == b"[DONE]"
    decoded = [json.loads(e) for e in events[:-1]]
    assert [e["text"] for e in decoded] == ["x", "xx", "xxx"]
    for event in decoded:
        assert event["meta_info"]["input_token_logprobs"] == [
            [-0.5, 1, None],
            [-0.25, 2, None],
        ]
    _assert_sessions_closed(sessions)


def _abort(rid, status_code=None):
    finish_reason = {"type": "abort", "message": "aborted"}
    if status_code is not None:
        finish_reason["status_code"] = status_code
    return {"text": "", "meta_info": {"id": rid, "finish_reason": finish_reason}}


def _finish(rid, finish_reason):
    return {"text": "", "meta_info": {"id": rid, "finish_reason": finish_reason}}


@pytest.mark.unit
@pytest.mark.parametrize(
    "body, expected",
    [
        (PREFILL_ERROR_BODY, "error_event"),
        (_sse({"error": {"message": "no code"}}), "error_event"),
        (PREFILL_ABORT_BODY, "aborted"),
        (_sse(_abort("r1", 400), _abort("r2", 500)), "aborted"),
        (_sse(_abort("r1")), "aborted"),
        # A request reports a null finish reason before its final one.
        (_sse(_finish("r1", None), _abort("r1", 400)), "aborted"),
        # Partially aborted batch: keep relaying decode for the survivor.
        (_sse(_abort("r1", 400), _finish("r2", {"type": "length"})), None),
        (PREFILL_FINISHED_BODY, None),
        (_sse(_finish("r1", None)), None),
        (b"data: [DONE]\n\n", None),
        (b"", None),
        (b"data: not json\n\n", None),
    ],
)
def test_prefill_stream_failure_classifier(body, expected):
    from sglang_router.mini_lb import _prefill_stream_failure

    assert _prefill_stream_failure(body) == expected


# --- #658 + #1706: a streaming decode also reports its failure in a 200 body ---

DECODE_ERROR_BODY = _sse(
    {
        "error": {
            "message": "The input is longer than the context length",
            "type": "invalid_request_error",
            "code": 400,
            "retryable": False,
        }
    }
)
DECODE_ABORT_BODY = _sse(_abort("r1", 400))


def _decode_answers(body):
    """A streaming decode that answers 200 with `body` at once, like a real
    engine rejecting a streaming request (error event, then `[DONE]`)."""
    return _prefill_answers(body)


def _prefill_waiting_for_decode(disconnected: asyncio.Event):
    """A streaming prefill that answers its headers at once, then sends nothing
    for SLOW_SECONDS (waiting for decode to ask for its KV); sets
    `disconnected` if MiniLB closes the connection first."""

    async def handler(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + SLOW_SECONDS
        try:
            while loop.time() < deadline:
                if request.transport is None or request.transport.is_closing():
                    disconnected.set()
                    return resp
                await asyncio.sleep(0.02)
            await resp.write(PREFILL_FINISHED_BODY)
            await resp.write_eof()
        except asyncio.CancelledError:
            disconnected.set()
            raise
        except ConnectionError:
            disconnected.set()
        return resp

    return handler


@pytest.mark.unit
@pytest.mark.parametrize("decode_body", [DECODE_ERROR_BODY, DECODE_ABORT_BODY])
def test_streaming_decode_inband_failure_does_not_wait_for_prefill(decode_body):
    """A streaming decode that rejects the request answers 200 with the failure
    in its body and never asks prefill for KV, so prefill never finishes (it
    waits for decode until its bootstrap timeout). Reading prefill's body before
    relaying decode would hang the client for that long; decode's stream must be
    relayed at once and prefill's connection closed so it aborts the request."""

    async def run():
        disconnected = asyncio.Event()
        observed = {}

        async def before_cleanup():
            observed["elapsed"] = time.monotonic() - start
            try:
                await asyncio.wait_for(disconnected.wait(), 1.0)
            except asyncio.TimeoutError:
                pass
            observed["disconnected"] = disconnected.is_set()

        start = time.monotonic()
        response, chunks, sessions = await _run(
            _prefill_waiting_for_decode(disconnected),
            _decode_answers(decode_body),
            before_cleanup=before_cleanup,
        )
        return response, chunks, sessions, observed

    response, chunks, sessions, observed = asyncio.run(run())
    assert response.status_code == 200
    assert b"".join(chunks) == decode_body
    assert observed["elapsed"] < EARLY_RETURN_BOUND_SECONDS, "MiniLB waited for prefill"
    assert observed["disconnected"], "MiniLB must close its connection to the prefill"
    _assert_sessions_closed(sessions)


@pytest.mark.unit
def test_decode_token_before_prefill_body_still_relays_decode():
    """Decode may emit its first token before prefill's 200 body has arrived.
    MiniLB must then still wait for prefill's body and relay decode in full,
    including the event it already read."""

    async def late_prefill(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await asyncio.sleep(0.3)
        await resp.write(PREFILL_FINISHED_BODY)
        await resp.write_eof()
        return resp

    response, chunks, sessions = asyncio.run(
        _run(late_prefill, _decode_answers(DECODE_BODY))
    )
    assert response.status_code == 200
    assert b"".join(chunks) == DECODE_BODY
    _assert_sessions_closed(sessions)


@pytest.mark.unit
def test_affinity_truncated_streaming_prefill_body_returns_502(monkeypatch):
    """A streaming prefill body lost before the stream is committed is a
    connection failure of that prefill, as for non-streaming bodies."""

    async def truncated(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b'data: {"text":')
        request.transport.close()
        return response

    async def waiting_decode(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await asyncio.sleep(5)
        return response

    async def run():
        async with _engine(truncated) as prefill, _engine(waiting_decode) as decode:
            balancer = _affinity_lb(prefill, decode)
            monkeypatch.setattr(mini_lb, "lb", balancer)
            response = await asyncio.wait_for(
                mini_lb._forward_to_backend(
                    {"text": "hi", "stream": True}, "generate", None
                ),
                timeout=2,
            )
            assert response.status_code == 502
            assert not balancer.affinity_policy.loads.is_reachable(prefill)
            assert balancer.affinity_policy.loads.is_reachable(decode)

    asyncio.run(run())
