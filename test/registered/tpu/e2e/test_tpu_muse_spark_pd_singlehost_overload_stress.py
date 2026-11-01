"""Muse Spark 1P1D overload stress with quality gates (#1867).

One TP8 host per engine: NVFP4 weights, FP8 KV, Raiden, multimodal, 256K
context, and a prefill prefix cache with host KV offload. The prefill pod runs
the mini-LB (session-affinity policy, which sheds with 429) and drives:

1. Overload: text (short prompts to planted-code prompts near 256K), image and
   video requests, while GSM8K and MMMU-Pro run through a proxy that retries
   every 429 until the item is answered.
2. Drain: stop new traffic and wait until both engines are idle.
3. The same quality evals again, on the drained system.

It fails on any wrong answer, timeout, connection error or non-429 HTTP error,
on a quality gate miss in either round, if the system never overloaded, if it
does not drain in time, or if the flow outlasts one hour after healthy.
"""

import json
import os
import socket
import threading
import time
import unittest
from typing import Any

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.pd_overload_stress import (
    FAILING_OUTCOMES,
    EngineLoadMonitor,
    LoadDriver,
    LoadDriverConfig,
    Outcome,
    RetryProxy,
    StressLedger,
    color_video_part,
    scrape_counters,
    square_image_part,
)
from sglang.test.tpu import MUSE_SPARK
from sglang.test.tpu.utils.pd_singlehost_base import PDSingleHostTestCase

register_tpu_ci(
    est_time=7200,
    suite="tpu-integration",
    nightly=True,
    labels=[
        "singlehost",
        "muse-spark-open-816b-a42b",
        "pd",
        "prefix-cache",
        "kv-offload",
        "multimodal",
        "video",
        "long-context",
        "stress",
    ],
)

TEST_NAME = "muse-spark-pd-singlehost-overload-stress"
MODEL = MUSE_SPARK["served_model_name"]

# The stress flow must finish within this long after the server is healthy.
FLOW_BUDGET_S = float(os.environ.get("STRESS_FLOW_BUDGET_S", "3600"))
# Keep the system overloaded at least this long, eval round 1 included.
MIN_OVERLOAD_S = float(os.environ.get("STRESS_MIN_OVERLOAD_S", "900"))
# Stop overload by this point even if a long-context size has not finished.
MAX_OVERLOAD_S = float(os.environ.get("STRESS_MAX_OVERLOAD_S", "1800"))
DRAIN_TIMEOUT_S = float(os.environ.get("STRESS_DRAIN_TIMEOUT_S", "600"))
# KV usage counts as idle at the pre-load baseline plus this slack.
IDLE_TOKEN_USAGE_SLACK = 0.02
LONG_CONTEXT_TOKENS = (32768, 131072, 253952)
CONCURRENCY = int(os.environ.get("STRESS_CONCURRENCY", "64"))

EVAL_GATES = [
    {
        "dataset": "gsm8k",
        "baseline": 0.98,
        "atol": 0.10,
        "num_examples": int(os.environ.get("STRESS_GSM8K_EXAMPLES", "100")),
        "num_threads": 16,
        "max_tokens": 4096,
        "temperature": 0.0,
        "reasoning_effort": "low",
    },
    {
        "dataset": "mmmu_pro_vision",
        "baseline": float(os.environ.get("STRESS_MMMU_BASELINE", "0.5")),
        "atol": float(os.environ.get("STRESS_MMMU_ATOL", "0.10")),
        "num_examples": int(os.environ.get("STRESS_MMMU_EXAMPLES", "100")),
        "num_threads": 16,
        "max_tokens": 4096,
        "temperature": 0.0,
        "reasoning_effort": "low",
    },
]

FAILURE_COUNTERS = (
    "num_aborted_requests_total",
    "num_bootstrap_failed_reqs_total",
    "num_transfer_failed_reqs_total",
)


def self_hostname() -> str:
    # External single-host roles drop TPU_WORKER_HOSTNAMES and set PD_SELF_HOST.
    if os.environ.get("PD_SELF_HOST"):
        return os.environ["PD_SELF_HOST"]
    hostnames = [h for h in os.environ.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    return hostnames[0] if hostnames else socket.gethostname()


class TestMuseSparkPDOverloadStress(PDSingleHostTestCase):
    config_module = "recipe/configs/muse_spark/pd_singlehost_overload_stress.py"
    log_path = "/tmp/pd_muse_stress_prefill.log"
    decode_log_path = "/tmp/pd_muse_stress_decode.log"
    self_hostname = staticmethod(self_hostname)
    tag = "pd-singlehost-muse-stress"

    @classmethod
    def build_model_config(cls) -> dict[str, Any]:
        return {"model": MODEL, "test_name": TEST_NAME, "gates": EVAL_GATES}

    def _engine_urls(self) -> dict[str, str]:
        prefill_port = int(os.environ.get("PREFILL_PORT", "30000"))
        decode_port = int(os.environ.get("DECODE_PORT", "30000"))
        return {
            "prefill": f"http://127.0.0.1:{prefill_port}",
            "decode": f"http://{self.decode_hostname()}:{decode_port}",
        }

    def _eval_round(self, name: str, proxy_url: str) -> tuple[bool, str]:
        from sglang.test.tpu import run_model_config

        started = time.monotonic()
        try:
            run_model_config(self.build_model_config(), base_url=proxy_url)
            passed, error = True, ""
        except AssertionError as failure:
            passed, error = False, str(failure)
        print(
            f"[{TEST_NAME}] eval {name}: {'PASS' if passed else 'FAIL'} "
            f"in {time.monotonic() - started:.0f}s {error}",
            flush=True,
        )
        return passed, error

    def _preflight(self) -> None:
        """One image and one video answered correctly before any load.

        A missing video decoder or vision tower fails every multimodal request;
        catching it here costs one request instead of the whole stress flow.
        """
        import requests

        cases = (
            (
                square_image_part("blue"),
                "What color is the square? Reply with the color only.",
                "blue",
            ),
            (
                color_video_part(("red", "green")),
                "Name the square colors at the beginning and end of the video, "
                "in order. Reply with only the two colors.",
                "red",
            ),
        )
        for part, question, expected in cases:
            response = requests.post(
                self.base_url + "/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": [part, {"type": "text", "text": question}],
                        }
                    ],
                    "temperature": 0,
                    "max_tokens": 512,
                    "chat_template_kwargs": {"reasoning_effort": "low"},
                },
                timeout=1800,
            )
            self.assertEqual(
                response.status_code,
                200,
                f"{part['type']} preflight failed: {response.text[:500]}",
            )
            answer = response.json()["choices"][0]["message"].get("content") or ""
            self.assertIn(
                expected, answer.lower(), f"{part['type']} preflight: {answer!r}"
            )

    def test_overload_drain_and_quality(self):
        if self.role == "decode":
            self.run_evaluations()
            return

        healthy_at = time.monotonic()
        self._preflight()
        failures: list[str] = []
        engines = self._engine_urls()
        driver_ledger = StressLedger()
        eval_ledger = StressLedger()
        monitor = EngineLoadMonitor(engines)
        idle_baseline = monitor.read()
        counters_before = {
            name: scrape_counters(url, FAILURE_COUNTERS)
            for name, url in engines.items()
        }
        proxy = RetryProxy(self.base_url, eval_ledger).start()
        driver = LoadDriver(
            self.base_url,
            LoadDriverConfig(
                model=MODEL,
                concurrency=CONCURRENCY,
                long_context_tokens=LONG_CONTEXT_TOKENS,
            ),
            driver_ledger,
        )
        tokens_per_line = driver.calibrate()
        print(f"[{TEST_NAME}] {tokens_per_line:.2f} prompt tokens per filler line")

        # Phase 1: overload, with eval round 1 inside it.
        monitor.start()
        driver.start()
        overload_started = time.monotonic()
        heartbeat_stop = threading.Event()

        def heartbeat() -> None:
            while not heartbeat_stop.wait(60):
                print(
                    f"[{TEST_NAME}] t={time.monotonic() - healthy_at:.0f}s "
                    f"driver={json.dumps(driver_ledger.summary()['by_kind'])} "
                    f"eval={json.dumps(eval_ledger.summary()['by_kind'])} "
                    f"eval_429_retries={eval_ledger.retried_429} "
                    f"peak={json.dumps({k: vars(v) for k, v in monitor.peak.items()})}",
                    flush=True,
                )

        threading.Thread(target=heartbeat, daemon=True).start()
        round1 = self._eval_round("round1-under-overload", proxy.url)
        long_kinds = [f"text_long_{t // 1024}k" for t in LONG_CONTEXT_TOKENS]
        while time.monotonic() - overload_started < MAX_OVERLOAD_S:
            long_done = all(
                driver_ledger.count(Outcome.OK, kind) > 0 for kind in long_kinds
            )
            if time.monotonic() - overload_started >= MIN_OVERLOAD_S and long_done:
                break
            time.sleep(10)
        overload_s = time.monotonic() - overload_started

        # Phase 2: stop new traffic and drain both engines.
        drain_started = time.monotonic()
        inflight_done = driver.stop(drain_timeout_s=DRAIN_TIMEOUT_S)
        monitor.stop()
        ceilings = {
            name: load.token_usage + IDLE_TOKEN_USAGE_SLACK
            for name, load in idle_baseline.items()
        }
        remaining = max(1.0, DRAIN_TIMEOUT_S - (time.monotonic() - drain_started))
        idle, after_drain = monitor.wait_idle(
            timeout_s=remaining, token_usage_ceiling=ceilings
        )
        drain_s = time.monotonic() - drain_started

        # Phase 3: the same evals on the drained system.
        round2 = self._eval_round("round2-after-drain", proxy.url)
        heartbeat_stop.set()
        proxy.stop()
        flow_s = time.monotonic() - healthy_at

        counters_after = {
            name: scrape_counters(url, FAILURE_COUNTERS)
            for name, url in engines.items()
        }

        # Verdict.
        if not round1[0]:
            failures.append(f"eval round 1 under overload failed: {round1[1]}")
        if not round2[0]:
            failures.append(f"eval round 2 after drain failed: {round2[1]}")
        for ledger_name, ledger in (("driver", driver_ledger), ("eval", eval_ledger)):
            for outcome in FAILING_OUTCOMES:
                n = ledger.count(outcome)
                if n:
                    failures.append(
                        f"{ledger_name}: {n} requests ended {outcome.value}"
                    )
        shed = driver_ledger.count(Outcome.SHED) + eval_ledger.retried_429
        peak = monitor.peak
        queued = max(p.waiting + p.pd_queued for p in peak.values())
        if shed == 0 and queued == 0:
            failures.append("system never overloaded: no 429 and no queued requests")
        for kind in long_kinds:
            if driver_ledger.count(Outcome.OK, kind) == 0:
                failures.append(f"no {kind} request completed with the right code")
        if not inflight_done:
            failures.append("in-flight requests did not finish within the drain limit")
        if not idle:
            failures.append(
                "engines did not drain to idle: "
                + json.dumps({k: vars(v) for k, v in after_drain.items()})
            )
        if flow_s > FLOW_BUDGET_S:
            failures.append(
                f"stress flow took {flow_s:.0f}s, over the {FLOW_BUDGET_S:.0f}s budget"
            )

        report = {
            "flow_s": round(flow_s),
            "overload_s": round(overload_s),
            "drain_s": round(drain_s),
            "rounds": {"round1": round1, "round2": round2},
            "driver": driver_ledger.summary(),
            "eval": eval_ledger.summary(),
            "peak_load": {k: vars(v) for k, v in peak.items()},
            "idle_baseline": {k: vars(v) for k, v in idle_baseline.items()},
            "after_drain": {k: vars(v) for k, v in after_drain.items()},
            "load_poll_errors": monitor.poll_errors,
            "failure_counter_deltas": {
                name: {
                    key: counters_after[name][key] - counters_before[name][key]
                    for key in FAILURE_COUNTERS
                }
                for name in engines
            },
            "failures": failures,
        }
        text = json.dumps(report, indent=1, default=str)
        print(f"[{TEST_NAME}] STRESS_REPORT\n{text}", flush=True)
        with open("/tmp/stress_report.json", "w") as out:
            out.write(text)
        self.assertEqual(failures, [], "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
