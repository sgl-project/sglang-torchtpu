"""Unit tests for the TPU P/D overload stress harness (no TPU, server, or model).

The Muse 1P1D overload nightly runs only on two TPU hosts, so the harness
contracts its verdict rests on are pinned here against an in-process fake router:

* a 429 is the router shedding load: the eval relay sends the request again and
  the eval client never sees it, while a 5xx or a 408 (the router's queue
  timeout) goes back to the client once and is recorded as a failure;
* every request queue the engine's metrics collector exports counts against the
  drain, a request in any of them keeps the engine busy, and an engine exporting
  no load gauges fails the check instead of reading idle;
* an answer that finishes without its planted code, or without the colors of
  the image or video it was sent in order, is corruption; one cut off at
  max_tokens before it answers proves nothing, except that a long-context probe
  must come back answered; a shed request is not a failure, and a shed probe is
  sent again until it is answered;
* Muse returns a reply cut off while it is still reasoning in reasoning_content
  alone, so the reasoning is graded when the content is empty, never over it;
* an engine failure counter is exported only once it first counts, and that
  first count still reads as growth;
* a sized prompt never exceeds its token budget, so the longest probe fits the
  context it is sized against.
"""

import base64
import importlib.util
import io
import json
import re
import threading
import time
import unittest
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import av
from PIL import Image, ImageColor

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.testing import overload_stress as stress

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_CODE = re.compile(r"The access code is ([A-Z0-9-]+)\.")


def _completion(
    text: str, finish_reason: str = "stop", reasoning: str | None = None
) -> dict[str, Any]:
    message = {"content": text}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10},
    }


def _user_text(body: dict[str, Any]) -> str:
    content = body["messages"][0]["content"]
    return " ".join(part.get("text", "") for part in content if part["type"] == "text")


def _media_colors(body: dict[str, Any]) -> list[str]:
    """The square colors of the request's image or video, decoded, in order."""
    part = body["messages"][0]["content"][0]
    data = base64.b64decode(part[part["type"]]["url"].split(",", 1)[1])
    if part["type"] == "image_url":
        frames = [Image.open(io.BytesIO(data)).convert("RGB")]
    else:
        with av.open(io.BytesIO(data)) as container:
            frames = [frame.to_image() for frame in container.decode(video=0)]
    colors: list[str] = []
    for frame in frames:
        center = frame.getpixel((frame.width // 2, frame.height // 2))
        color = min(
            stress._COLORS,
            key=lambda name: sum(
                (a - b) ** 2 for a, b in zip(ImageColor.getrgb(name), center)
            ),
        )
        if not colors or colors[-1] != color:
            colors.append(color)
    return colors


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


class TestEngineFailureCounters(CustomTestCase):
    def test_failure_counters_are_ones_the_engines_export(self):
        # A misspelled counter is never exported, so it would read zero and pass
        # every run; the collector is the list of names.
        spec = importlib.util.find_spec("sglang.srt.observability.metrics_collector")
        exported = set(
            re.findall(r'name="(sglang:[a-z_]+)"', Path(spec.origin).read_text())
        )
        self.assertLessEqual(set(stress.FAILURE_COUNTERS), exported)

    def test_a_counter_missing_until_its_first_count_reports_that_count(self):
        before = stress.failure_counts_from_metrics(_metrics_text())
        after = stress.failure_counts_from_metrics(
            _metrics_text()
            + 'sglang:num_transfer_failed_reqs_total{tp_rank="0"} 1.0\n'
            + 'sglang:num_transfer_failed_reqs_total{tp_rank="1"} 2.0\n'
        )

        self.assertEqual(
            stress.failure_count_growth(before, after),
            {"sglang:num_transfer_failed_reqs_total": 3.0},
        )
        self.assertEqual(stress.failure_count_growth(after, after), {})


class TestAnswerGrading(CustomTestCase):
    def test_a_color_counts_only_as_a_whole_word(self):
        # Reasoning text is full of words like "considered" that contain one.
        self.assertTrue(stress.answer_names_colors("I considered it red.", ("red",)))
        self.assertFalse(stress.answer_names_colors("I considered it.", ("red",)))


class TestOverloadDriver(CustomTestCase):
    SIZER = stress.PromptSizer(tokens_per_line=17.25, overhead_tokens=50)
    TEXT_ONLY = stress.LoadMix(
        text=1.0,
        image=0.0,
        video=0.0,
        text_prompt_tokens=(512,),
        text_prompt_weights=(1.0,),
    )
    MEDIA_ONLY = stress.LoadMix(
        text=0.0,
        image=1.0,
        video=1.0,
        text_prompt_tokens=(512,),
        text_prompt_weights=(1.0,),
    )

    def _run(
        self,
        router: _FakeRouter,
        *,
        mix: stress.LoadMix | None = None,
        probes: tuple[int, ...] = (),
        burst_size: int = 0,
    ) -> stress.OverloadDriver:
        self.addCleanup(router.close)
        driver = stress.OverloadDriver(
            router.url,
            model="m",
            sizer=self.SIZER,
            mix=mix or self.TEXT_ONLY,
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

    def test_reasoning_is_graded_only_when_the_content_is_empty(self):
        # A probe cut off while reasoning had its code there and failed as
        # truncated; reasoning must still not excuse a wrong content answer.
        def replying(content: str, finish_reason: str) -> _FakeRouter:
            def respond(count: int, body: dict) -> tuple[int, dict]:
                code = _CODE.search(_user_text(body)).group(1)
                return 200, _completion(
                    content, finish_reason, reasoning=f"The code is {code}."
                )

            return _FakeRouter(respond)

        driver = self._run(replying("", "length"), probes=(2048,))
        self.assertEqual(driver.failures(), [])
        self.assertEqual(driver.probe_results[2048].category, stress.OK)

        failures = self._run(replying("ZZZZ-ZZZZ", "stop")).failures()
        self.assertEqual(len(failures), 1)
        self.assertIn("text/wrong_answer", failures[0])

    def test_image_and_video_answers_must_name_their_colors_in_order(self):
        cases = {
            "in order": (
                lambda colors: _completion(", ".join(colors)),
                {"image/ok", "video/ok"},
            ),
            "reversed": (
                lambda colors: _completion(", ".join(reversed(colors))),
                {"image/ok", "video/wrong_answer"},
            ),
            "cut off": (
                lambda colors: _completion("", "length"),
                {"image/truncated", "video/truncated"},
            ),
        }
        for name, (answer, outcomes) in cases.items():
            with self.subTest(answer=name):
                driver = self._run(
                    _FakeRouter(
                        lambda count, body, answer=answer: (
                            200,
                            answer(_media_colors(body)),
                        )
                    ),
                    mix=self.MEDIA_ONLY,
                )
                self.assertEqual(set(driver.summary()["counts"]), outcomes)

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
