"""The mini load balancer must return an upstream rejection with its own
status instead of wrapping it in a 200 event stream, and must close its
upstream session however a request ends."""

import asyncio
import json
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web
from sglang_router import mini_lb
from sglang_router.mini_lb import MiniLoadBalancer


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
        # A completed stream lets the prefill finish in the background.
        await asyncio.gather(*list(mini_lb._background_tasks))
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


async def _decode_dies_mid_stream(request):
    """Streams two chunks, then drops the connection without ending the
    chunked body, as a decode whose pod is preempted does."""
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    for i in range(2):
        await resp.write(json.dumps({"text": "x" * (i + 1)}).encode() + b"\n\n")
    await asyncio.sleep(0.05)
    request.transport.close()
    return resp


@pytest.mark.unit
def test_decode_dying_mid_stream_reports_an_error(caplog):
    with caplog.at_level("ERROR", logger="sglang_router.mini_lb"):
        response, chunks, sessions = asyncio.run(_run(_stream, _decode_dies_mid_stream))
    assert response.status_code == 200  # already committed when it broke
    body = b"".join(chunks)
    assert body.count(b'"text"') == 2
    events = [e for e in body.split(b"\n\n") if e.startswith(b"data: {")]
    error = json.loads(events[-1][len(b"data: ") :])["error"]
    assert error["code"] == 502
    assert error["retryable"] is True
    assert "stream broke" in error["message"]
    assert body.endswith(b"data: [DONE]\n\n")
    records = [r for r in caplog.records if r.name == "sglang_router.mini_lb"]
    assert len(records) == 1 and records[0].exc_info is None  # one line
    _assert_sessions_closed(sessions)


def _slow_prefill(finish_after_s, outcome):
    """A prefill that answers its first chunk at once but ends its own
    response only `finish_after_s` later (its KV transfer is still being
    wrapped up). Records whether it got to finish or saw a disconnect."""

    async def handler(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b'data: {"text": "x"}\n\n')
        try:
            await asyncio.sleep(finish_after_s)
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            outcome.append("finished")
        except (ConnectionResetError, asyncio.CancelledError):
            outcome.append("disconnected")
            raise
        return resp

    return handler


@pytest.mark.unit
def test_completed_stream_lets_the_prefill_finish():
    outcome = []
    response, chunks, sessions = asyncio.run(_run(_slow_prefill(0.3, outcome), _stream))
    assert response.status_code == 200
    assert b"".join(chunks).endswith(b"data: [DONE]\n\n")
    # The prefill was not cut off once the decode had answered.
    assert outcome == ["finished"]
    _assert_sessions_closed(sessions)


@pytest.mark.unit
def test_stuck_prefill_is_closed_after_the_drain_timeout(monkeypatch):
    monkeypatch.setattr(mini_lb, "PREFILL_DRAIN_TIMEOUT_SECS", 0.2)
    outcome = []
    response, chunks, sessions = asyncio.run(_run(_slow_prefill(2, outcome), _stream))
    assert response.status_code == 200
    assert b"".join(chunks).endswith(b"data: [DONE]\n\n")
    _assert_sessions_closed(sessions)
