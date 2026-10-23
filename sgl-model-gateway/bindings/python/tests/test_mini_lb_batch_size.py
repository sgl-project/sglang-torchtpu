"""MiniLB must not crash while shaping the PD bootstrap fields of a /generate
request whose prompt is empty or malformed: it forwards the request and
returns the engines' own rejection instead of a bare 500."""

import asyncio

import orjson
import pytest
from aiohttp import web
from sglang_router import mini_lb

EMPTY_INPUT_IDS_ERROR = {"error": {"message": "input_ids cannot be empty."}}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("request_data", "expected"),
    [
        ({"input_ids": []}, None),
        ({"input_ids": [], "session_params": {"id": "s"}}, None),
        ({"input_ids": [1, 2, 3]}, None),
        ({"input_ids": [[1, 2], [3]]}, 2),
        ({"input_ids": [[]]}, 1),
        ({"input_ids": [[1, 2], []]}, 2),
        ({"input_ids": 5}, None),
        ({"input_ids": "abc"}, None),
        ({"text": "hello"}, None),
        ({"text": ["a", "b"]}, 2),
        ({"text": []}, 0),
        ({"text": 5}, None),
        ({}, None),
    ],
)
def test_get_request_batch_size(request_data, expected):
    assert mini_lb._get_request_batch_size(request_data) == expected


async def _serve(handler):
    app = web.Application()
    app.router.add_post("/generate", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def _engine(received):
    """Fake engine /generate that records each request and, like
    GenerateReqInput.normalize_batch_and_arguments, rejects an empty
    input_ids list with a 400 unless a session supplies the prompt."""

    async def handler(request):
        body = await request.json()
        received.append(body)
        if body.get("input_ids") == [] and not body.get("session_params"):
            return web.json_response(EMPTY_INPUT_IDS_ERROR, status=400)
        return web.json_response({"text": "ok", "meta_info": {"dp_rank": 0}})

    return handler


async def _generate(monkeypatch, request_data, test_external_dp_routing=False):
    received = []
    prefill_runner, prefill_url = await _serve(_engine(received))
    decode_runner, decode_url = await _serve(_engine(received))
    try:
        lb = mini_lb.MiniLoadBalancer.__new__(mini_lb.MiniLoadBalancer)
        lb.timeout = 10
        lb.prefill_urls = [prefill_url]
        lb.prefill_bootstrap_ports = [8998]
        lb.decode_urls = [decode_url]
        lb.test_external_dp_routing = test_external_dp_routing
        lb.prefill_dp_size = 1
        lb.decode_dp_size = 1
        monkeypatch.setattr(mini_lb, "lb", lb)
        response = await mini_lb.handle_generate_request(request_data)
        return response, received
    finally:
        await prefill_runner.cleanup()
        await decode_runner.cleanup()


@pytest.mark.unit
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("test_external_dp_routing", [False, True])
def test_empty_input_ids_returns_engine_rejection(
    monkeypatch, stream, test_external_dp_routing
):
    request_data = {"input_ids": [], "stream": stream}
    response, received = asyncio.run(
        _generate(monkeypatch, request_data, test_external_dp_routing)
    )
    assert response.status_code == 400
    assert orjson.loads(response.body) == EMPTY_INPUT_IDS_ERROR
    # Forwarded to both engines as a single request with scalar bootstrap fields.
    assert len(received) == 2
    assert all(isinstance(body["bootstrap_room"], int) for body in received)


@pytest.mark.unit
def test_empty_input_ids_with_session_is_forwarded(monkeypatch):
    request_data = {"input_ids": [], "session_params": {"id": "s"}}
    response, received = asyncio.run(_generate(monkeypatch, request_data))
    assert response.status_code == 200
    assert len(received) == 2
    assert all(isinstance(body["bootstrap_room"], int) for body in received)


@pytest.mark.unit
@pytest.mark.parametrize(("input_ids", "batch_size"), [([[]], 1), ([[1, 2], []], 2)])
def test_batch_with_empty_item_gets_per_item_bootstrap(
    monkeypatch, input_ids, batch_size
):
    response, received = asyncio.run(_generate(monkeypatch, {"input_ids": input_ids}))
    assert response.status_code == 200
    for body in received:
        assert len(body["bootstrap_host"]) == batch_size
        assert len(body["bootstrap_port"]) == batch_size
        assert len(body["bootstrap_room"]) == batch_size
    # Prefill and decode receive the same rooms so their KV transfer pairs up.
    assert received[0]["bootstrap_room"] == received[1]["bootstrap_room"]
