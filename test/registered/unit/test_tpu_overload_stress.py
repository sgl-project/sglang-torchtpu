"""Unit tests for the TPU P/D overload stress harness (no TPU, server, or model).

The Muse 1P1D overload nightly runs only on two TPU hosts, so the harness
contracts its verdict rests on are pinned here against an in-process fake router:

* a 429 is the router shedding load: the eval relay sends the request again and
  the eval client never sees it, while a 5xx or a 408 (the router's queue
  timeout) goes back to the client once and is recorded as a failure;
* every request queue the engine's metrics collector exports counts against the
  drain, a request in any of them keeps the engine busy, and an engine exporting
  no load gauges fails the check instead of reading idle;
* an answer that finishes without its planted code is corruption; one cut off at
  max_tokens before the code proves nothing, except that a long-context probe
  must come back answered; a shed request is not a failure, and a shed probe is
  sent again until it is answered;
* a sized prompt never exceeds its token budget, so the longest probe fits the
  context it is sized against.
"""

import importlib.util
import json
import re
import threading
import time
import unittest
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.testing import overload_stress as stress

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_CODE = re.compile(r"The access code is ([A-Z0-9-]+)\.")


def _completion(text: str, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10},
    }


def _user_text(body: dict[str, Any]) -> str:
    content = body["messages"][0]["content"]
    return " ".join(part.get("text", "") for part in content if part["type"] == "text")


class _FakeRouter:
    """In-process HTTP server answering each POST from a scripted function."""

    def __init__(self, respond: Callable[[int, dict], tuple[int, dict]]):
        self.respond = respond
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        fake = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with fake._lock:
                    fake.requests.append(body)
                    count = len(fake.requests)
                status, payload = fake.respond(count, body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _metrics_text(queued: dict[str, int] | None = None) -> str:
    lines = [
        'sglang:num_running_reqs{tp_rank="0"} 0.0',
        'sglang:token_usage{tp_rank="0"} 0.0',
    ]
    lines += [
        f'{name}{{tp_rank="0"}} {(queued or {}).get(name, 0)}'
        for name in stress.QUEUE_GAUGES
    ]
    return "\n".join(lines) + "\n"


class TestEvalRetryRelay(CustomTestCase):
    def _relay(self, router: _FakeRouter) -> stress.EvalRetryRelay:
        self.addCleanup(router.close)
        relay = stress.EvalRetryRelay(
            router.url, retry_budget_s=30, request_timeout_s=10
        ).start()
        self.addCleanup(relay.stop)
        return relay

    def _ask(self, relay: stress.EvalRetryRelay) -> stress.ChatResult:
        return stress.post_chat(
            relay.url,
            model="m",
            content=[{"type": "text", "text": "question"}],
            max_tokens=8,
            timeout=30,
        )

    def test_shed_eval_request_is_retried_until_answered(self):
        router = _FakeRouter(
            lambda count, body: (
                (429, {"error": "busy"}) if count <= 2 else (200, _completion("42"))
            )
        )
        relay = self._relay(router)

        result = self._ask(relay)

        self.assertEqual((result.category, result.text), (stress.OK, "42"))
        self.assertEqual(relay.retried_429, 2)
        self.assertEqual(relay.failures, [])

    def test_errors_reach_the_client_once_and_are_recorded(self):
        for status, category in ((503, stress.SERVER_ERROR), (408, stress.TIMEOUT)):
            with self.subTest(status=status):
                router = _FakeRouter(
                    lambda count, body, status=status: (
                        (status, {"error": "x"})
                        if count == 1
                        else (200, _completion("late"))
                    )
                )
                relay = self._relay(router)

                result = self._ask(relay)

                self.assertEqual(result.status, status)
                self.assertEqual(len(router.requests), 1)
                self.assertEqual(len(relay.failures), 1)
                self.assertTrue(relay.failures[0].startswith(category))


class TestEngineDrain(CustomTestCase):
    def test_drain_counts_every_queue_gauge_the_engine_exports(self):
        # A request held in a queue the drain check does not read would let a
        # stuck engine pass as drained; the collector is the list of queues.
        spec = importlib.util.find_spec("sglang.srt.observability.metrics_collector")
        exported = set(
            re.findall(
                r'"(sglang:num_[a-z_]*queue_reqs)"', Path(spec.origin).read_text()
            )
        )
        self.assertIn("sglang:num_queue_reqs", exported)
        self.assertEqual(set(stress.QUEUE_GAUGES), exported)

    def test_a_request_in_any_queue_keeps_the_engine_busy(self):
        self.assertTrue(
            stress.engine_load_from_metrics("e", _metrics_text()).is_idle(0.01)
        )
        for gauge in stress.QUEUE_GAUGES:
            with self.subTest(gauge=gauge):
                load = stress.engine_load_from_metrics(
                    "e", _metrics_text(queued={gauge: 1})
                )
                self.assertFalse(load.is_idle(0.01))

    def test_engine_without_load_gauges_fails_instead_of_reading_idle(self):
        with self.assertRaisesRegex(ValueError, "enable-metrics"):
            stress.engine_load_from_metrics("e", "# HELP unrelated\nother_metric 0\n")


class TestOverloadDriver(CustomTestCase):
    SIZER = stress.PromptSizer(tokens_per_line=17.25, overhead_tokens=50)
    TEXT_ONLY = stress.LoadMix(
        text=1.0,
        image=0.0,
        video=0.0,
        text_prompt_tokens=(512,),
        text_prompt_weights=(1.0,),
    )

    def _run(
        self,
        router: _FakeRouter,
        *,
        probes: tuple[int, ...] = (),
        burst_size: int = 0,
    ) -> stress.OverloadDriver:
        self.addCleanup(router.close)
        driver = stress.OverloadDriver(
            router.url,
            model="m",
            sizer=self.SIZER,
            mix=self.TEXT_ONLY,
            clients=2,
            probe_prompt_tokens=probes,
            max_tokens=16,
            probe_max_tokens=16,
            request_timeout_s=10,
            probe_deadline_s=30,
            burst_size=burst_size,
            burst_interval_s=0.05,
        ).start()

        def bursted() -> bool:
            return any(key.startswith("burst/") for key in driver.summary()["counts"])

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not (
            len(router.requests) >= 12 and (burst_size == 0 or bursted())
        ):
            time.sleep(0.05)
        driver.stop_new_traffic()
        self.assertTrue(driver.join(30))
        return driver

    def test_a_finished_answer_without_its_code_is_corruption(self):
        driver = self._run(
            _FakeRouter(lambda count, body: (200, _completion("ZZZZ-ZZZZ")))
        )

        failures = driver.failures()

        self.assertEqual(len(failures), 1)
        self.assertIn("text/wrong_answer", failures[0])

    def test_truncation_is_inconclusive_for_load_but_fails_a_probe(self):
        driver = self._run(
            _FakeRouter(lambda count, body: (200, _completion("Let me", "length"))),
            probes=(2048,),
        )

        failures = driver.failures()

        self.assertEqual(len(failures), 1)
        self.assertIn("2048-token probe ended truncated", failures[0])
        self.assertGreater(driver.summary()["counts"]["text/truncated"], 0)

    def test_shed_requests_pass_and_a_shed_probe_is_sent_again(self):
        seen_probes = set()
        lock = threading.Lock()

        def respond(count: int, body: dict) -> tuple[int, dict]:
            text = _user_text(body)
            if text.startswith("Document "):
                with lock:
                    first = text not in seen_probes
                    seen_probes.add(text)
                if first:
                    return 429, {"error": "busy"}
            elif count % 2:
                return 429, {"error": "busy"}
            return 200, _completion(_CODE.search(text).group(1))

        driver = self._run(_FakeRouter(respond), probes=(2048,), burst_size=4)

        self.assertEqual(driver.failures(), [])
        self.assertGreater(driver.shed_count(), 0)
        self.assertEqual(driver.probe_results[2048].category, stress.OK)
        self.assertTrue(
            any(key.startswith("burst/") for key in driver.summary()["counts"])
        )


class TestPromptSizer(CustomTestCase):
    def test_sized_prompts_stay_within_their_token_budget(self):
        # 17.25 tokens per filler line over a 44-token frame, as measured.
        sizer = stress.PromptSizer.from_measurements((256, 2304), (4460, 39788))
        for target in (512, 8192, 259072):
            with self.subTest(target=target):
                lines = sizer.lines_for(target)
                self.assertLessEqual(
                    sizer.overhead_tokens + lines * sizer.tokens_per_line,
                    target - sizer.margin_tokens,
                )


if __name__ == "__main__":
    unittest.main()
