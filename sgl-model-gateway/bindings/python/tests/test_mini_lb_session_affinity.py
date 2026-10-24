"""MiniLB's session_affinity policy: sticky prefill with load-gated spillover,
sampled load-based decode and guardrail 429s."""

import asyncio
import json
import random
import socket

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from sglang_router import mini_lb
from sglang_router.launch_router import parse_router_args
from sglang_router.mini_lb_policy import (
    DECODE_POOL,
    POLICY_NO_AFFINITY,
    POLICY_OVERLOADED,
    POLICY_STICKY,
    PREFILL_POOL,
    EngineLoads,
    EngineOverloaded,
    LoadPoller,
    SessionAffinityPolicy,
)
from sglang_router.router_args import RouterArgs

P1, P2 = "http://p1:30000", "http://p2:30000"
D1, D2 = "http://d1:30000", "http://d2:30000"


def _policy():
    return SessionAffinityPolicy(rng=random.Random(0))


def _set_load(policy, url, **metrics):
    base = {
        "num_waiting_reqs": 0,
        "num_running_reqs": 0,
        "num_waiting_uncached_tokens": 0,
    }
    base.update(metrics)
    policy.loads.update(url, base)


def _select(policy, routing_key, prefill_urls=(P1, P2), decode_urls=(D1, D2)):
    return policy.select(
        prefill_urls=list(prefill_urls),
        decode_urls=list(decode_urls),
        routing_key=routing_key,
    )


@pytest.mark.unit
def test_new_key_goes_to_less_loaded_prefill_and_then_sticks():
    policy = _policy()
    _set_load(policy, P1, num_waiting_reqs=4)
    _set_load(policy, P2, num_waiting_reqs=1)

    prefill, _ = _select(policy, "s-1")
    assert (prefill.engine, prefill.policy) == (P2, POLICY_NO_AFFINITY)

    # Sticky while the pinned load stays within the gate, even when not least loaded.
    _set_load(policy, P1, num_waiting_reqs=4)
    _set_load(policy, P2, num_waiting_reqs=10)
    prefill, _ = _select(policy, "s-1")
    assert (prefill.engine, prefill.policy) == (P2, POLICY_STICKY)


@pytest.mark.unit
def test_overloaded_pin_spills_and_repins():
    policy = _policy()
    _select(policy, "s-1", prefill_urls=(P1,))
    _set_load(policy, P1, num_waiting_reqs=7)
    _set_load(policy, P2, num_waiting_reqs=2)

    prefill, _ = _select(policy, "s-1")
    assert (prefill.engine, prefill.policy) == (P2, POLICY_OVERLOADED)
    assert prefill.rejection == "primary:load"

    prefill, _ = _select(policy, "s-1")
    assert (prefill.engine, prefill.policy) == (P2, POLICY_STICKY)


@pytest.mark.unit
def test_gate_ignores_imbalance_below_min_load():
    policy = _policy()
    _select(policy, "s-1", prefill_urls=(P1,))
    _set_load(policy, P1, num_waiting_reqs=5)

    prefill, _ = _select(policy, "s-1")
    assert (prefill.engine, prefill.policy) == (P1, POLICY_STICKY)


@pytest.mark.unit
def test_pinned_engine_over_guardrail_spills():
    policy = _policy()
    _select(policy, "s-1", prefill_urls=(P1,))
    _set_load(policy, P1, num_waiting_uncached_tokens=2_000_001)

    prefill, _ = _select(policy, "s-1")
    assert (prefill.engine, prefill.policy) == (P2, POLICY_OVERLOADED)
    assert prefill.rejection == "guardrail:num_waiting_uncached_tokens"


@pytest.mark.unit
def test_prefill_429_when_spill_target_is_also_over_guardrail():
    policy = _policy()
    _select(policy, "s-1", prefill_urls=(P1,))
    _set_load(policy, P1, num_waiting_uncached_tokens=2_000_001)
    _set_load(policy, P2, num_waiting_uncached_tokens=3_000_000)

    with pytest.raises(EngineOverloaded) as raised:
        _select(policy, "s-1")
    assert str(raised.value) == (
        f"engine {P2} overloaded on num_waiting_uncached_tokens: load=3000000"
    )


@pytest.mark.unit
def test_decode_picks_fewer_running_and_rejects_over_guardrail():
    policy = _policy()
    _set_load(policy, D1, num_running_reqs=9)
    _set_load(policy, D2, num_running_reqs=3)
    _, decode = _select(policy, None)
    assert (decode.engine, decode.policy) == (D2, POLICY_NO_AFFINITY)

    _set_load(policy, D1, num_running_reqs=1, num_waiting_reqs=25)
    _set_load(policy, D2, num_running_reqs=5, num_waiting_reqs=30)
    with pytest.raises(
        EngineOverloaded, match="overloaded on num_waiting_reqs: load=25"
    ):
        _select(policy, "s-1")
    # A request rejected at decode leaves its key unpinned.
    _set_load(policy, D1)
    _set_load(policy, D2)
    _set_load(policy, P1, num_waiting_reqs=1)
    prefill, _ = _select(policy, "s-1")
    assert prefill.policy == POLICY_NO_AFFINITY


@pytest.mark.unit
def test_request_without_key_is_never_pinned():
    policy = _policy()
    for _ in range(5):
        prefill, _ = _select(policy, None)
        assert prefill.policy == POLICY_NO_AFFINITY


@pytest.mark.unit
@pytest.mark.parametrize("waiting_per_rank", [12, 13])
def test_load_poller_sums_dp_ranks_and_marks_dead_engines(waiting_per_rank, caplog):
    async def handler(request):
        assert request.query["include"] == "core,queues"
        rank = {"num_waiting_reqs": waiting_per_rank, "num_running_reqs": 5}
        return web.json_response({"loads": [rank, rank]})

    async def run():
        app = web.Application()
        app.router.add_get("/v1/loads", handler)
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            dead_url = f"http://127.0.0.1:{reserved.getsockname()[1]}"
            async with TestServer(app) as server:
                live_url = str(server.make_url("")).rstrip("/")
                loads = EngineLoads()
                poller = LoadPoller(
                    loads=loads,
                    metric_names=frozenset({"num_waiting_reqs", "num_running_reqs"}),
                    interval_secs=0.05,
                    timeout_secs=1,
                )
                async with aiohttp.ClientSession() as session:
                    await poller._poll(session=session, url=live_url)
                    for _ in range(2):
                        await poller._poll(session=session, url=dead_url)
                assert loads.get(live_url, "num_waiting_reqs") == 2 * waiting_per_rank
                assert loads.get(live_url, "num_running_reqs") == 10
                assert loads.is_reachable(live_url)
                assert not loads.is_reachable(dead_url)
                policy = SessionAffinityPolicy(loads=loads)
                if waiting_per_rank == 12:
                    prefill, decode = _select(
                        policy, "s-1", [dead_url, live_url], [dead_url, live_url]
                    )
                    assert prefill.engine == decode.engine == live_url
                else:
                    with pytest.raises(
                        EngineOverloaded, match="num_waiting_reqs: load=26"
                    ):
                        _select(policy, None, [live_url], [dead_url, live_url])

    asyncio.run(run())
    assert sum("is unreachable:" in record.message for record in caplog.records) == 1


def _lb(prefill_urls, decode_urls):
    args = RouterArgs(
        host="127.0.0.1",
        port=0,
        policy="random",
        mini_lb_policy="session_affinity",
        pd_disaggregation=True,
        prefill_urls=[(url, 8998) for url in prefill_urls],
        decode_urls=list(decode_urls),
        test_external_dp_routing=False,
        request_timeout_secs=10,
    )
    return mini_lb.MiniLoadBalancer(args)


@pytest.mark.unit
def test_overloaded_request_gets_429(monkeypatch):
    load_balancer = _lb([P1], [D1])
    load_balancer.affinity_policy.loads.update(
        D1, {"num_running_reqs": 0, "num_waiting_reqs": 25}
    )
    monkeypatch.setattr(mini_lb, "lb", load_balancer)

    response = asyncio.run(
        mini_lb._forward_to_backend({"text": "hello"}, "generate", "s-1")
    )
    assert response.status_code == 429
    assert json.loads(response.body) == {
        "error": {"message": f"engine {D1} overloaded on num_waiting_reqs: load=25"}
    }


@pytest.mark.unit
def test_routing_key_header_reaches_the_policy(monkeypatch):
    load_balancer = _lb([P1], [D1])
    monkeypatch.setattr(mini_lb, "lb", load_balancer)
    # Header propagation does not need background network polling.
    monkeypatch.setattr(load_balancer, "start_load_polling", lambda: None)
    seen_keys = []

    def select(*, prefill_urls, decode_urls, routing_key):
        seen_keys.append(routing_key)
        raise EngineOverloaded(engine=D1, metric="num_waiting_reqs", load=25)

    monkeypatch.setattr(load_balancer.affinity_policy, "select", select)

    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    with TestClient(mini_lb.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": []},
            headers={"x-routing-key": "s-42"},
        )
    assert response.status_code == 429
    assert seen_keys == ["s-42"]


@pytest.mark.unit
def test_cli_flag_selects_policy():
    args = parse_router_args(
        [
            "--mini-lb",
            "--mini-lb-policy",
            "session_affinity",
            "--pd-disaggregation",
            "--prefill",
            P1,
            "--decode",
            D1,
        ]
    )
    assert args.mini_lb_policy == "session_affinity"
    assert mini_lb.MiniLoadBalancer(args).affinity_policy is not None
    assert parse_router_args(["--mini-lb"]).mini_lb_policy == "random"
    random_args = parse_router_args(["--mini-lb", "--mini-lb-policy", "random"])
    assert random_args.mini_lb_policy == "random"


@pytest.mark.unit
def test_default_random_ignores_routing_key(monkeypatch):
    args = RouterArgs(
        pd_disaggregation=True,
        prefill_urls=[(P1, 8998), (P2, 8999)],
        decode_urls=[D1, D2],
    )
    load_balancer = mini_lb.MiniLoadBalancer(args)
    assert load_balancer.affinity_policy is None
    picks = iter((0, 0, 1, 1, 0, 0))
    monkeypatch.setattr(mini_lb.random, "randint", lambda _low, _high: next(picks))
    assert load_balancer.select_pair(routing_key="s-1") == (P1, 8998, D1)
    assert load_balancer.select_pair(routing_key="s-1") == (P2, 8999, D2)
    assert load_balancer.select_pair(routing_key="s-1") == (P1, 8998, D1)


@pytest.mark.unit
@pytest.mark.parametrize("sample_sizes", [None, (1, 1), (3, 5), (16, 12)])
def test_pool_sample_sizes_preserve_guardrails(sample_sizes):
    prefill_urls = [f"http://p{i}:30000" for i in range(10)]
    decode_urls = [f"http://d{i}:30000" for i in range(10)]
    pool_overrides = {}
    if sample_sizes is not None:
        pool_overrides = {
            "prefill_pool": PREFILL_POOL._replace(
                num_reference_engines=sample_sizes[0]
            ),
            "decode_pool": DECODE_POOL._replace(num_reference_engines=sample_sizes[1]),
        }
    policy = SessionAffinityPolicy(rng=random.Random(0), **pool_overrides)
    for i, url in enumerate(prefill_urls):
        _set_load(policy, url, num_waiting_reqs=i, num_running_reqs=20 - i)
    for i, url in enumerate(decode_urls):
        _set_load(policy, url, num_running_reqs=i, num_waiting_reqs=20 - i)
    # The lowest-load engines violate guardrails.
    _set_load(policy, prefill_urls[0], num_waiting_uncached_tokens=2_000_001)
    _set_load(policy, decode_urls[0], num_waiting_reqs=25)

    prefill, decode = _select(policy, None, prefill_urls, decode_urls)
    for decision, urls, sample_size in zip(
        (prefill, decode), (prefill_urls, decode_urls), sample_sizes or (8, 8)
    ):
        assert len(decision.sampled_references) == min(sample_size, len(urls))
        passing = {ref.url for ref in decision.sampled_references} - {urls[0]}
        assert decision.engine == min(passing, key=urls.index)


@pytest.mark.unit
@pytest.mark.parametrize("pool", ["prefill", "decode"])
@pytest.mark.parametrize("sample_size", [0, -1])
def test_nonpositive_sample_size_rejected(pool, sample_size):
    config = (PREFILL_POOL if pool == "prefill" else DECODE_POOL)._replace(
        num_reference_engines=sample_size
    )
    with pytest.raises(ValueError, match=f"{pool} sample size must be at least 1"):
        SessionAffinityPolicy(**{f"{pool}_pool": config})


@pytest.mark.unit
def test_unreachable_pin_is_replaced():
    policy = _policy()
    _select(policy, "s-1", prefill_urls=(P1,))
    policy.loads.mark_unreachable(P1)
    prefill, _ = _select(policy, "s-1")
    assert (prefill.engine, prefill.policy) == (P2, POLICY_NO_AFFINITY)
    prefill, _ = _select(policy, "s-1")
    assert (prefill.engine, prefill.policy) == (P2, POLICY_STICKY)


@pytest.mark.unit
def test_removed_pin_is_replaced():
    """Removing a pinned prefill must move the key to a registered engine."""
    policy = _policy()
    _select(policy, "s-1", prefill_urls=(P1,))
    prefill, _ = _select(policy, "s-1", prefill_urls=(P2,))
    assert (prefill.engine, prefill.policy) == (P2, POLICY_NO_AFFINITY)
    prefill, _ = _select(policy, "s-1", prefill_urls=(P1, P2))
    assert (prefill.engine, prefill.policy) == (P2, POLICY_STICKY)


@pytest.mark.unit
def test_affinity_pool_falls_back_when_all_engines_unreachable(caplog):
    """An entirely excluded pool falls back once per outage instead of crashing."""
    policy = _policy()
    policy.loads.mark_unreachable(P1)
    policy.loads.mark_unreachable(P2)
    for _ in range(2):
        prefill, _ = _select(policy, None)
        assert prefill.engine in (P1, P2)
    assert (
        sum("pool prefill fail-open" in record.message for record in caplog.records)
        == 1
    )
    _set_load(policy, P2)
    prefill, _ = _select(policy, None)
    assert prefill.engine == P2
    policy.loads.mark_unreachable(P2)
    _select(policy, None)
    assert (
        sum("pool prefill fail-open" in record.message for record in caplog.records)
        == 2
    )


@pytest.mark.unit
def test_poll_timeouts_exclude_engine_until_success(caplog):
    """Repeated silent polls must stop an idle-looking engine winning placement."""

    async def run():
        respond = False
        release = asyncio.Event()

        async def handler(_request):
            if not respond:
                await release.wait()
            return web.json_response({"loads": [{"num_running_reqs": 2}]})

        app = web.Application()
        app.router.add_get("/v1/loads", handler)
        async with TestServer(app) as server:
            url = str(server.make_url("")).rstrip("/")
            loads = EngineLoads()
            poller = LoadPoller(
                loads=loads,
                metric_names=frozenset({"num_running_reqs"}),
                interval_secs=0.01,
                timeout_secs=0.02,
            )
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=0.02)
            ) as session:
                try:
                    for _ in range(2):
                        await poller._poll(session=session, url=url)
                        assert loads.is_reachable(url)
                    for _ in range(2):
                        await poller._poll(session=session, url=url)
                        assert not loads.is_reachable(url)
                    respond = True
                    await poller._poll(session=session, url=url)
                    assert loads.is_reachable(url)
                    assert loads.get(url, "num_running_reqs") == 2
                    respond = False
                    for _ in range(2):
                        await poller._poll(session=session, url=url)
                        assert loads.is_reachable(url)
                finally:
                    release.set()

    caplog.set_level("INFO", logger="sglang_router.mini_lb_policy")
    asyncio.run(run())
    messages = [record.message for record in caplog.records]
    assert sum("is unreachable after" in message for message in messages) == 1
    assert sum("is reachable again" in message for message in messages) == 1


@pytest.mark.unit
@pytest.mark.parametrize("malformed", ["invalid_json", "missing_metric", "wrong_type"])
def test_malformed_loads_keep_snapshot_and_warn_once(caplog, malformed):
    """Bad load payloads retain valid counts without flooding the router log."""

    async def run():
        valid = False

        async def handler(_request):
            if valid:
                return web.json_response({"loads": [{"num_running_reqs": 7}]})
            if malformed == "invalid_json":
                return web.Response(text="{", content_type="application/json")
            if malformed == "missing_metric":
                return web.json_response({"loads": [{}]})
            return web.json_response({"loads": [{"num_running_reqs": "invalid"}]})

        app = web.Application()
        app.router.add_get("/v1/loads", handler)
        async with TestServer(app) as server:
            url = str(server.make_url("")).rstrip("/")
            loads = EngineLoads()
            loads.update(url, {"num_running_reqs": 5})
            poller = LoadPoller(
                loads=loads,
                metric_names=frozenset({"num_running_reqs"}),
                interval_secs=0.01,
                timeout_secs=1,
            )
            async with aiohttp.ClientSession() as session:
                for _ in range(2):
                    await poller._poll(session=session, url=url)
                assert loads.get(url, "num_running_reqs") == 5
                assert (
                    sum(
                        "malformed /v1/loads" in record.message
                        for record in caplog.records
                    )
                    == 1
                )
                valid = True
                await poller._poll(session=session, url=url)
                assert loads.get(url, "num_running_reqs") == 7
                valid = False
                await poller._poll(session=session, url=url)
                assert loads.get(url, "num_running_reqs") == 7
                assert (
                    sum(
                        "malformed /v1/loads" in record.message
                        for record in caplog.records
                    )
                    == 2
                )

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize("stream", [False, True])
def test_empty_batch_bootstrap_arrays_stay_aligned(monkeypatch, stream):
    """An empty batch must not acquire a bootstrap room for a nonexistent item."""
    balancer = _lb([P1], [D1])
    monkeypatch.setattr(mini_lb, "lb", balancer)

    async def upstream(request, *_args, **_kwargs):
        return mini_lb.ORJSONResponse(
            {
                name: request[name]
                for name in ("bootstrap_host", "bootstrap_port", "bootstrap_room")
            }
        )

    monkeypatch.setattr(balancer, "generate_stream" if stream else "generate", upstream)
    response = asyncio.run(
        mini_lb._forward_to_backend({"text": [], "stream": stream}, "generate", "s-1")
    )
    assert json.loads(response.body) == {
        "bootstrap_host": [],
        "bootstrap_port": [],
        "bootstrap_room": [],
    }
