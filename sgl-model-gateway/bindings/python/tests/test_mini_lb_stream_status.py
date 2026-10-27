"""The mini load balancer must return an upstream rejection with its own
status instead of wrapping it in a 200 event stream, and must close its
upstream session however a request ends."""

import asyncio
import contextlib
import json
import socket
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
    )
    return MiniLoadBalancer(args)


async def _run(prefill_handler, decode_handler, max_chunks=None):
    """Issue one streaming request through MiniLB and read up to
    `max_chunks` of its body; returns the response, the chunks read and the
    upstream sessions MiniLB opened."""
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
            {"text": "hello", "stream": True}, prefill_url, decode_url
        )
        chunks = []
        if hasattr(response, "body_iterator"):
            async for chunk in response.body_iterator:
                chunks.append(chunk)
                if max_chunks is not None and len(chunks) >= max_chunks:
                    await response.body_iterator.aclose()
                    break
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
