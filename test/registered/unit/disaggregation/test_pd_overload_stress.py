"""CPU tests for the P/D overload stress helpers (#1867)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import asyncio
import random
import threading
import unittest

import requests
from aiohttp import web

from sglang.test.pd_overload_stress import (
    EngineLoad,
    EngineLoadMonitor,
    LoadDriver,
    LoadDriverConfig,
    Outcome,
    RequestRecord,
    RetryProxy,
    StressLedger,
    TrafficMix,
    _answer_matches,
    classify_status,
    parse_loads,
    planted_code_prompt,
)
from sglang.test.test_utils import CustomTestCase


class _FakeLB:
    """A local HTTP server that answers with a scripted status sequence."""

    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = 0
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self._ready.wait(10)

    def _serve(self):
        asyncio.set_event_loop(self._loop)

        async def handle(request):
            self.calls += 1
            status = self.statuses.pop(0) if self.statuses else 200
            body = await request.json()
            return web.json_response({"echo": body, "status": status}, status=status)

        async def start():
            app = web.Application()
            app.router.add_post("/v1/chat/completions", handle)
            self._runner = web.AppRunner(app)
            await self._runner.setup()
            site = web.TCPSite(self._runner, "127.0.0.1", 0)
            await site.start()
            self.port = self._runner.addresses[0][1]
            self._ready.set()

        self._loop.run_until_complete(start())
        self._loop.run_forever()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        async def stop():
            await self._runner.cleanup()

        asyncio.run_coroutine_threadsafe(stop(), self._loop).result(10)
        self._loop.call_soon_threadsafe(self._loop.stop)


class TestRetryProxy(CustomTestCase):
    def _proxy(self, lb, ledger, **kwargs):
        proxy = RetryProxy(
            lb.url, ledger, initial_backoff_s=0.01, max_backoff_s=0.02, **kwargs
        ).start()
        self.addCleanup(proxy.stop)
        self.addCleanup(lb.close)
        return proxy

    def test_shed_requests_are_retried_until_answered(self):
        lb = _FakeLB([429, 429, 429, 200])
        ledger = StressLedger()
        proxy = self._proxy(lb, ledger)

        response = requests.post(
            proxy.url + "/v1/chat/completions", json={"q": 1}, timeout=10
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["echo"], {"q": 1})
        self.assertEqual(lb.calls, 4)
        self.assertEqual(ledger.retried_429, 3)
        self.assertEqual(ledger.count(Outcome.OK, "eval"), 1)

    def test_server_errors_pass_through_and_are_recorded(self):
        lb = _FakeLB([503])
        ledger = StressLedger()
        proxy = self._proxy(lb, ledger)

        response = requests.post(
            proxy.url + "/v1/chat/completions", json={}, timeout=10
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(lb.calls, 1)
        self.assertEqual(ledger.count(Outcome.SERVER_ERROR, "eval"), 1)
        self.assertEqual(ledger.failures()[0].status, 503)

    def test_shed_past_the_deadline_counts_as_unanswered(self):
        lb = _FakeLB([429] * 1000)
        ledger = StressLedger()
        proxy = self._proxy(lb, ledger, retry_deadline_s=0.2)

        response = requests.post(
            proxy.url + "/v1/chat/completions", json={}, timeout=10
        )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(ledger.count(Outcome.CLIENT_ERROR, "eval"), 1)
        self.assertEqual(ledger.count(Outcome.OK, "eval"), 0)


class TestLoadDriverBackoff(CustomTestCase):
    def test_shed_workers_back_off_instead_of_hammering(self):
        lb = _FakeLB([429] * 100000)
        self.addCleanup(lb.close)
        ledger = StressLedger()
        driver = LoadDriver(
            lb.url,
            LoadDriverConfig(
                model="m",
                concurrency=2,
                mix=TrafficMix(text_short=1.0, image=0, video=0, text_long=0),
                shed_backoff_s=(0.25, 0.25),
            ),
            ledger,
        )
        driver._tokens_per_line = 20.0  # skip calibration against the fake

        driver.start()
        import time

        time.sleep(1.0)
        driver.stop(drain_timeout_s=5)

        # Two workers, 0.25 s apart after each 429: at most ~10 requests in 1 s.
        self.assertGreater(lb.calls, 0)
        self.assertLessEqual(lb.calls, 12)
        self.assertEqual(ledger.count(Outcome.SHED), lb.calls)


class TestClassification(CustomTestCase):
    def test_status_classes(self):
        self.assertEqual(
            [classify_status(s) for s in (200, 429, 400, 500, 503)],
            [
                Outcome.OK,
                Outcome.SHED,
                Outcome.CLIENT_ERROR,
                Outcome.SERVER_ERROR,
                Outcome.SERVER_ERROR,
            ],
        )

    def test_answer_must_contain_expected_tokens_in_order(self):
        self.assertTrue(_answer_matches("the code is 12345678.", ("12345678",)))
        self.assertTrue(_answer_matches("red, then blue", ("red", "blue")))
        self.assertFalse(_answer_matches("blue, then red", ("red", "blue")))
        self.assertFalse(_answer_matches("the code is 1234567", ("12345678",)))

    def test_ledger_counts_and_keeps_only_failure_examples(self):
        ledger = StressLedger(max_failure_examples=1)
        ledger.add(RequestRecord("image", Outcome.OK, 200, 1.0, prompt_tokens=300))
        ledger.add(RequestRecord("image", Outcome.SHED, 429, 0.1))
        ledger.add(RequestRecord("video", Outcome.TIMEOUT, None, 9.0, "t1"))
        ledger.add(RequestRecord("video", Outcome.TIMEOUT, None, 9.0, "t2"))

        self.assertEqual(ledger.count(Outcome.SHED), 1)
        self.assertEqual(ledger.count(Outcome.TIMEOUT, "video"), 2)
        self.assertEqual([r.detail for r in ledger.failures()], ["t1"])
        self.assertEqual(ledger.max_prompt_tokens("image"), 300)
        summary = ledger.summary()
        self.assertEqual(summary["by_kind"]["image"], {"ok": 1, "shed_429": 1})


class TestPlantedPrompt(CustomTestCase):
    def test_one_code_planted_and_asked_for(self):
        prompt = planted_code_prompt(500, random.Random(7))

        self.assertEqual(len(prompt.code), 8)
        self.assertEqual(prompt.text.count(prompt.code), 1)
        self.assertIn(f"vault {prompt.vault} is {prompt.code}.", prompt.text)
        self.assertTrue(prompt.text.rstrip().endswith("eight-digit code."))
        self.assertEqual(prompt.text.count("Ledger entry"), 500)

    def test_prompts_differ_across_requests(self):
        rng = random.Random(1)
        first, second = planted_code_prompt(50, rng), planted_code_prompt(50, rng)
        self.assertNotEqual(first.code, second.code)
        self.assertNotEqual(first.text.splitlines()[0], second.text.splitlines()[0])


def _loads(running, waiting, prealloc=0, usage=0.0):
    return {
        "loads": [
            {
                "dp_rank": 0,
                "num_running_reqs": running,
                "num_waiting_reqs": waiting,
                "token_usage": usage,
                "disaggregation": {
                    "mode": "decode",
                    "decode_prealloc_queue_reqs": prealloc,
                },
            }
        ]
    }


class TestEngineLoadMonitor(CustomTestCase):
    def test_parse_sums_ranks_and_pd_queues(self):
        payload = _loads(2, 3, prealloc=4, usage=0.5)
        payload["loads"].append(
            {"dp_rank": 1, "num_running_reqs": 1, "token_usage": 0.7}
        )

        self.assertEqual(parse_loads(payload), EngineLoad(3, 3, 4, 0.7))

    def test_wait_idle_needs_settled_idle_reads_under_the_ceiling(self):
        script = [
            _loads(4, 9, usage=0.9),
            _loads(0, 0, usage=0.30),  # idle but KV not released yet
            _loads(0, 0, usage=0.11),
            _loads(0, 0, usage=0.10),
            _loads(0, 0, usage=0.10),
        ]
        monitor = EngineLoadMonitor(
            {"decode": "http://d"}, fetch=lambda url: script.pop(0)
        )

        idle, loads = monitor.wait_idle(
            timeout_s=100,
            token_usage_ceiling={"decode": 0.12},
            settle_polls=3,
            sleep=lambda s: None,
        )

        self.assertTrue(idle)
        self.assertEqual(loads["decode"], EngineLoad(0, 0, 0, 0.10))
        self.assertEqual(monitor.peak["decode"], EngineLoad(4, 9, 0, 0.9))

    def test_wait_idle_times_out_while_requests_remain(self):
        now = [0.0]

        def clock():
            now[0] += 1.0
            return now[0]

        monitor = EngineLoadMonitor(
            {"prefill": "http://p"}, fetch=lambda url: _loads(1, 0, usage=0.0)
        )

        idle, loads = monitor.wait_idle(
            timeout_s=5,
            token_usage_ceiling={"prefill": 1.0},
            sleep=lambda s: None,
            clock=clock,
        )

        self.assertFalse(idle)
        self.assertEqual(loads["prefill"].running, 1)


if __name__ == "__main__":
    unittest.main()
