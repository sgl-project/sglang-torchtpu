"""MiniLB must stop routing to a backend whose /health_generate fails, route to
it again once it recovers, and still route when a whole role is unhealthy."""

import asyncio
import logging
import socket

import pytest
from aiohttp import web
from sglang_router import mini_lb
from sglang_router.router_args import RouterArgs

# Enough random picks that missing a selectable backend has probability < 1e-9.
_PICKS = 64


class _Backends:
    """Real HTTP backends whose /health_generate status a test can change."""

    def __init__(self):
        self.status = {}
        self.probes = {}
        self.probe_started = {}
        # Backends whose probe hangs until `release` is set.
        self.hung = set()
        self.release = asyncio.Event()
        self._runners = []

    async def start(self, count):
        return [await self._start_one() for _ in range(count)]

    async def _start_one(self):
        url = None

        async def health_generate(_request):
            self.probes[url] += 1
            self.probe_started[url].set()
            if url in self.hung:
                await self.release.wait()
            return web.Response(status=self.status[url])

        app = web.Application()
        app.router.add_get("/health_generate", health_generate)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        self.status[url] = 200
        self.probes[url] = 0
        self.probe_started[url] = asyncio.Event()
        self._runners.append(runner)
        return url

    async def close(self):
        self.release.set()
        for runner in self._runners:
            await runner.cleanup()


def _refused_url():
    """URL of a local port that nothing listens on."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def _lb(prefill_urls, decode_urls, policy="random"):
    return mini_lb.MiniLoadBalancer(
        RouterArgs(
            mini_lb_policy=policy,
            pd_disaggregation=True,
            prefill_urls=[(url, 8998 + i) for i, url in enumerate(prefill_urls)],
            decode_urls=list(decode_urls),
            request_timeout_secs=10,
        )
    )


def _selected(load_balancer):
    """Prefill and decode URLs chosen over many select_pair calls."""
    prefills, decodes = set(), set()
    for _ in range(_PICKS):
        prefill, _, decode = load_balancer.select_pair()
        prefills.add(prefill)
        decodes.add(decode)
    return prefills, decodes


def _run(test, *, num_prefills=1, num_decodes=2):
    async def run():
        backends = _Backends()
        try:
            prefill_urls = await backends.start(num_prefills)
            decode_urls = await backends.start(num_decodes)
            await test(backends, prefill_urls, decode_urls)
        finally:
            await backends.close()

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize("policy", ["random", "session_affinity"])
def test_all_healthy_routes_to_every_backend(policy):
    async def test(_backends, prefill_urls, decode_urls):
        load_balancer = _lb(prefill_urls, decode_urls, policy)
        assert _selected(load_balancer) == (set(prefill_urls), set(decode_urls))

        await load_balancer.probe_backend_health()

        assert _selected(load_balancer) == (set(prefill_urls), set(decode_urls))

    _run(test, num_prefills=2)


@pytest.mark.unit
@pytest.mark.parametrize("policy", ["random", "session_affinity"])
@pytest.mark.parametrize("failure", [503, 500, "refused"])
def test_unhealthy_decode_is_skipped_until_it_recovers(policy, failure):
    async def test(backends, prefill_urls, decode_urls):
        healthy, sick = decode_urls
        if failure == "refused":
            sick = _refused_url()
            decode_urls = [healthy, sick]
        else:
            backends.status[sick] = failure
        load_balancer = _lb(prefill_urls, decode_urls, policy)

        await load_balancer.probe_backend_health()
        assert _selected(load_balancer) == (set(prefill_urls), {healthy})

        if failure == "refused":
            return
        backends.status[sick] = 200
        await load_balancer.probe_backend_health()
        assert _selected(load_balancer) == (set(prefill_urls), {healthy, sick})

    _run(test)


@pytest.mark.unit
def test_unhealthy_prefill_is_skipped_with_its_bootstrap_port():
    async def test(backends, prefill_urls, decode_urls):
        backends.status[prefill_urls[0]] = 503
        load_balancer = _lb(prefill_urls, decode_urls)

        await load_balancer.probe_backend_health()

        pairs = {load_balancer.select_pair()[:2] for _ in range(_PICKS)}
        assert pairs == {(prefill_urls[1], 8999)}

    _run(test, num_prefills=2, num_decodes=1)


@pytest.mark.unit
def test_hung_backend_is_unhealthy_after_probe_timeout(monkeypatch):
    monkeypatch.setattr(mini_lb, "HEALTH_CHECK_TIMEOUT_SECS", 0.2)

    async def test(backends, prefill_urls, decode_urls):
        healthy, hung = decode_urls
        backends.hung.add(hung)
        load_balancer = _lb(prefill_urls, decode_urls)

        await load_balancer.probe_backend_health()

        assert _selected(load_balancer)[1] == {healthy}

    _run(test)


@pytest.mark.unit
def test_all_decodes_unhealthy_routes_to_all_and_warns_once(caplog):
    async def test(backends, prefill_urls, decode_urls):
        for url in decode_urls:
            backends.status[url] = 503
        load_balancer = _lb(prefill_urls, decode_urls)

        with caplog.at_level(logging.WARNING, logger=mini_lb.logger.name):
            await load_balancer.probe_backend_health()
            assert _selected(load_balancer)[1] == set(decode_urls)
            await load_balancer.probe_backend_health()
            assert _selected(load_balancer)[1] == set(decode_urls)
        warnings = [r for r in caplog.records if "no healthy decode" in r.message]
        assert len(warnings) == 1

        # Once one decode recovers, routing leaves the panic floor.
        backends.status[decode_urls[0]] = 200
        await load_balancer.probe_backend_health()
        assert _selected(load_balancer)[1] == {decode_urls[0]}

        # A second transition into the panic floor warns again.
        backends.status[decode_urls[0]] = 503
        with caplog.at_level(logging.WARNING, logger=mini_lb.logger.name):
            await load_balancer.probe_backend_health()
            assert _selected(load_balancer)[1] == set(decode_urls)
        warnings = [r for r in caplog.records if "no healthy decode" in r.message]
        assert len(warnings) == 2

    _run(test)


@pytest.mark.unit
def test_role_switch_starts_worker_healthy_under_new_role():
    async def test(backends, prefill_urls, decode_urls):
        healthy, switched = decode_urls
        backends.status[switched] = 503
        load_balancer = _lb(prefill_urls, decode_urls)
        await load_balancer.probe_backend_health()
        assert _selected(load_balancer)[1] == {healthy}

        load_balancer.apply_role_switch(switched, "prefill", 9000)

        prefills = {load_balancer.select_pair()[:2] for _ in range(_PICKS)}
        assert prefills == {(prefill_urls[0], 8998), (switched, 9000)}
        assert _selected(load_balancer)[1] == {healthy}

    _run(test)


@pytest.mark.unit
def test_removed_worker_comes_back_healthy():
    async def test(backends, prefill_urls, decode_urls):
        healthy, sick = decode_urls
        backends.status[sick] = 503
        load_balancer = _lb(prefill_urls, decode_urls)
        await load_balancer.probe_backend_health()

        load_balancer.remove_worker(sick)
        assert _selected(load_balancer)[1] == {healthy}
        load_balancer.add_worker(sick, "decode")

        assert _selected(load_balancer)[1] == {healthy, sick}

    _run(test)


@pytest.mark.unit
def test_probe_result_for_a_worker_removed_mid_probe_is_dropped():
    async def test(backends, prefill_urls, decode_urls):
        healthy, sick = decode_urls
        backends.status[sick] = 503
        backends.hung.add(sick)
        load_balancer = _lb(prefill_urls, decode_urls)

        probe = asyncio.create_task(load_balancer.probe_backend_health())
        await backends.probe_started[sick].wait()
        load_balancer.remove_worker(sick)
        load_balancer.add_worker(sick, "decode")
        backends.release.set()
        await probe

        # The re-added worker is unknown, so it starts healthy.
        assert _selected(load_balancer)[1] == {healthy, sick}

    _run(test)


async def _wait_for(predicate):
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached in 2 s")


@pytest.mark.unit
def test_health_polling_runs_in_lifespan_and_stops_cleanly(monkeypatch):
    monkeypatch.setenv("SGLANG_MINI_LB_HEALTH_POLL_INTERVAL_SECS", "0.01")

    async def test(backends, prefill_urls, decode_urls):
        healthy, sick = decode_urls
        backends.status[sick] = 503
        load_balancer = _lb(prefill_urls, decode_urls)
        monkeypatch.setattr(mini_lb, "lb", load_balancer)
        tasks_before = asyncio.all_tasks()

        async with mini_lb._lifespan(mini_lb.app):
            await _wait_for(lambda: _selected(load_balancer)[1] == {healthy})
            backends.status[sick] = 200
            await _wait_for(lambda: _selected(load_balancer)[1] == {healthy, sick})
            assert backends.probes[sick] >= 2

        assert asyncio.all_tasks() == tasks_before

    _run(test)


@pytest.mark.unit
def test_health_polling_survives_an_unexpected_probe_error(monkeypatch):
    monkeypatch.setenv("SGLANG_MINI_LB_HEALTH_POLL_INTERVAL_SECS", "0.01")

    async def test(backends, prefill_urls, decode_urls):
        healthy, sick = decode_urls
        backends.status[sick] = 503
        load_balancer = _lb(prefill_urls, decode_urls)
        real_probe = load_balancer.probe_backend_health
        calls = []

        async def flaky_probe():
            calls.append(None)
            if len(calls) == 1:
                raise RuntimeError("boom")
            await real_probe()

        monkeypatch.setattr(load_balancer, "probe_backend_health", flaky_probe)
        polling = load_balancer.start_health_polling()
        try:
            await _wait_for(lambda: _selected(load_balancer)[1] == {healthy})
        finally:
            polling.cancel()
            await asyncio.gather(polling, return_exceptions=True)

    _run(test)


@pytest.mark.unit
@pytest.mark.parametrize("interval", ["0", "-1"])
def test_non_positive_interval_disables_health_polling(monkeypatch, interval):
    monkeypatch.setenv("SGLANG_MINI_LB_HEALTH_POLL_INTERVAL_SECS", interval)

    async def test(backends, prefill_urls, decode_urls):
        load_balancer = _lb(prefill_urls, decode_urls)
        assert load_balancer.start_health_polling() is None
        assert sum(backends.probes.values()) == 0

    _run(test)


@pytest.mark.unit
@pytest.mark.parametrize("interval", ["abc", "nan"])
def test_invalid_interval_is_rejected(monkeypatch, interval):
    monkeypatch.setenv("SGLANG_MINI_LB_HEALTH_POLL_INTERVAL_SECS", interval)
    with pytest.raises(ValueError, match="SGLANG_MINI_LB_HEALTH_POLL_INTERVAL_SECS"):
        _lb(["http://p:1"], ["http://d:1"])
