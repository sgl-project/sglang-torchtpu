"""Muse Spark 1P1D overload stress with quality gates, one TPU host per engine.

Prefill and decode each own one tpu7x 2x2x1 host (TP8) and exchange fp8 KV over
Raiden; the prefill engine keeps a prefix cache with host KV offload. The test
overloads the pair with text, image and video traffic while GSM8K and MMMU-Pro
run, stops the traffic, waits for both engines to drain, and runs both evals
again. It passes when:

* both eval rounds clear their floors with every item answered: an eval request
  the router sheds with a 429 is retried, never scored;
* the overload holds while the first round runs (429s, or requests queued on an
  engine) and is shed with 429s only: no 5xx, timeouts or connection errors;
* every text answer that finishes names its planted code, and the long-context
  probes, up to the full 256K context, all complete with theirs;
* both engines drain to idle (nothing queued or running, KV usage back at its
  pre-load level) within the drain limit, and no process died;
* the flow ends within an hour of the pair becoming healthy.

A P/D engine does not shed load (max_queued_requests applies only without
disaggregation), so the prefill pod starts its engine on its own and fronts the
pair with the Rust router under an admission cap, not the mini load balancer.

The nightly runs it on the external cluster (external-topology.yaml), where the
two engines share one 2x2x2 slice.
"""

import argparse
import http.client
import json
import os
import socket
import subprocess
import sys
import time
import unittest
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import RecipeLauncher, run_model_config, wait_for_health
from sglang.test.tpu.utils.pd_singlehost_base import PDSingleHostTestCase, peer_hostname
from sglang_tpu.testing.overload_stress import (
    OK,
    EvalRetryRelay,
    LoadMix,
    MetricsMonitor,
    OverloadDriver,
    measure_prompt_sizer,
    post_chat,
    scrape_engine,
    wait_until_idle,
)

register_tpu_ci(
    # Both engines cold-start in parallel, then up to an hour of stress flow.
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
        "seq-along-lane",
    ],
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 4))
BENCH = os.path.join(REPO, "benchmark", "tpu")
RENDER_EXTERNAL_JOBSET = os.path.join(
    REPO, ".github", "tpu-ci", "external", "render_external_jobset.py"
)

LAUNCH_CONFIG_MODULE = "recipe/configs/muse_spark/pd_singlehost_overload.py"

E = os.environ.get

# Must match the recipe.
MODEL_PATH = "/weights/Muse-Spark-Open-816B-A42B-NVFP4"
SERVED_MODEL_NAME = "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"
CONTEXT_LENGTH = 262144
KV_CACHE_DTYPE = "fp8_e4m3"
DISAGG_TRANSFER_BACKEND = "raiden"
RETRACTION_BACKUP = "cpu_tensor"

PREFILL_ROLE = "prefill"
DECODE_ROLE = "decode"
SELF_HOST_ENV = "PD_SELF_HOST"
DECODE_HOST_ENV = "PD_DECODE_HOST"

# A v7x host has four physical chips and eight addressable TPU devices.
HOSTS_PER_ENGINE = 1
TP_SIZE = 8

# The roles live in different pods, so their HTTP ports may match.
ENGINE_PORT = 30000
BOOTSTRAP_PORT = 8998
ROUTER_PORT = int(E("LB_PORT", "8000"))
LOCAL_HOST = "127.0.0.1"

WAIT_SECS = int(E("WAIT_SECS", "3600"))
HEALTH_TIMEOUT = float(E("HEALTH_TIMEOUT", "3600"))

PREFILL_LOG = "/tmp/pd_muse_overload_prefill.log"
DECODE_LOG = "/tmp/pd_muse_overload_decode.log"
ROUTER_LOG = "/tmp/pd_muse_overload_router.log"

# Router admission. The token bucket holds ROUTER_MAX_CONCURRENT tokens and
# refills --rate-limit-tokens-per-second on top of the tokens finished requests
# return (0 falls back to the cap), so it limits the admission rate, not how many
# requests are in flight: closed-loop clients are all admitted within seconds.
# Bursts of BURST_SIZE concurrent requests, more than a full bucket holds, keep
# it shedding while the evals run. A zero-length queue turns every request the
# bucket cannot admit into an immediate 429: a queued request could instead
# expire into a 408, which the gate counts as a timeout.
ROUTER_MAX_CONCURRENT = 16
ROUTER_REFILL_PER_S = 1
BURST_SIZE = 2 * ROUTER_MAX_CONCURRENT
BURST_INTERVAL_S = 15

# Closed-loop load: the backlog never exceeds the clients, one burst, the
# probes and the eval threads.
LOAD_CLIENTS = 12
LOAD_MIX = LoadMix(
    text=2.0,
    image=1.0,
    video=1.0,
    text_prompt_tokens=(512, 2048, 8192, 32768),
    text_prompt_weights=(4.0, 3.0, 2.0, 1.0),
)
# Muse reasons before it answers, so the budget leaves room for both; an answer
# that still runs out is truncated rather than graded.
LOAD_MAX_TOKENS = 2048
PROBE_MAX_TOKENS = 2048
# Up to the full context, less the probe's answer and a margin for the template.
PROBE_PROMPT_TOKENS = (65536, 131072, CONTEXT_LENGTH - PROBE_MAX_TOKENS - 2048)
REQUEST_TIMEOUT_S = 1800
CHAT_TEMPLATE_KWARGS = {"reasoning_effort": "low"}

OVERLOAD_WARMUP_S = 60
DRAIN_TIMEOUT_S = 900
# From both engines turning healthy to the end of the second round. The DFlash
# 1P1D nightly's pair took 32 min to turn healthy on the same hosts, so this
# also keeps the run inside the P/D JobSet's 8700 s deadline.
FLOW_LIMIT_S = 3600
# token_usage excludes evictable prefix cache, so an idle engine reads its
# pre-load level; the margin absorbs the gauge's two-decimal rounding.
IDLE_USAGE_MARGIN = 0.01

# The engines run 8 requests at a time and serve the queue first come, first
# served. Sixteen eval threads, about as many requests as the load keeps in
# flight, hold the evals to a fair share of the engines, and 64 items per eval
# keep both rounds well inside FLOW_LIMIT_S.
EVAL_THREADS = 16
EVAL_EXAMPLES = 64
# Full TPU calibration, 2026-10-24: 97.65% on all 1,319 questions
# (test_tpu_muse_spark_gsm8k.py).
GSM8K_GATE = {
    "dataset": "gsm8k",
    "baseline": 0.9765,
    "atol": 0.10,
    "num_examples": EVAL_EXAMPLES,
    "num_threads": EVAL_THREADS,
    "max_tokens": 4096,
    "temperature": 0.0,
    "reasoning_effort": "low",
}
# Uncalibrated: Muse has no TPU MMMU-Pro score and its card is gated. The floor
# sits well above the 10-option chance rate (0.10), so a broken image path fails
# while the model's real score is still unknown; raise it once a run measures it.
MMMU_PRO_GATE = {
    "dataset": "mmmu_pro_vision",
    "baseline": 0.25,
    "atol": 0.0,
    "num_examples": EVAL_EXAMPLES,
    "num_threads": EVAL_THREADS,
    "max_tokens": 4096,
    "temperature": 0.0,
    "reasoning_effort": "low",
}


def validate_model_source(model_path: str) -> None:
    """Fail clearly for a missing mounted checkpoint."""
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"checkpoint incomplete or missing at {model_path}; "
            "provision it on the weights volume"
        )


def self_hostname(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    if values.get(SELF_HOST_ENV):
        return values[SELF_HOST_ENV]
    hostnames = [h for h in values.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    return hostnames[0] if hostnames else socket.gethostname()


def _get_json(url: str, timeout: int = 30) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def engine_configuration_failures(
    *, prefill_info: Mapping[str, Any], decode_info: Mapping[str, Any]
) -> list[str]:
    expected = {
        "disaggregation_transfer_backend": DISAGG_TRANSFER_BACKEND,
        "tp_size": TP_SIZE,
        "nnodes": HOSTS_PER_ENGINE,
        "kv_cache_dtype": KV_CACHE_DTYPE,
        "context_length": CONTEXT_LENGTH,
        "enable_multimodal": True,
        "enable_metrics": True,
    }
    failures = []
    for role, info in ((PREFILL_ROLE, prefill_info), (DECODE_ROLE, decode_info)):
        for key, value in {**expected, "disaggregation_mode": role}.items():
            if info.get(key) != value:
                failures.append(
                    f"{role} reports {key}={info.get(key)!r}, not {value!r}"
                )
    if not prefill_info.get("enable_hierarchical_cache"):
        failures.append("prefill runs without host KV offload")
    if prefill_info.get("disable_radix_cache"):
        failures.append("prefill runs without a prefix cache")
    if decode_info.get("disaggregation_decode_retraction_backup") != RETRACTION_BACKUP:
        failures.append(
            "decode reports disaggregation_decode_retraction_backup="
            f"{decode_info.get('disaggregation_decode_retraction_backup')!r}, "
            f"not {RETRACTION_BACKUP!r}"
        )
    return failures


def router_command(prefill_url: str, decode_url: str) -> list[str]:
    """The Rust P/D router, capped so the overflow is shed with 429s.

    Retries and the circuit breaker are off so a failing engine answers the
    client directly, instead of being retried away or turned into router 503s.
    The health check gets 30s: an overloaded engine's /health answers late.
    """
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from benchmark.tpu.launch_server import python_command

    return [
        *python_command("sglang_router.launch_router"),
        "--pd-disaggregation",
        "--prefill",
        prefill_url,
        str(BOOTSTRAP_PORT),
        "--decode",
        decode_url,
        "--host",
        "0.0.0.0",
        "--port",
        str(ROUTER_PORT),
        "--max-concurrent-requests",
        str(ROUTER_MAX_CONCURRENT),
        "--rate-limit-tokens-per-second",
        str(ROUTER_REFILL_PER_S),
        "--queue-size",
        "0",
        "--request-timeout-secs",
        str(REQUEST_TIMEOUT_S),
        "--health-check-timeout-secs",
        "30",
        "--disable-retries",
        "--disable-circuit-breaker",
    ]


def wait_for_first_completion(
    base_url: str, timeout: float, router_process: subprocess.Popen
) -> None:
    """The router's /health runs no P/D generation; one completion does."""
    deadline = time.monotonic() + timeout
    while True:
        if router_process.poll() is not None:
            raise RuntimeError(f"router exited with code {router_process.returncode}")
        result = post_chat(
            base_url,
            model=SERVED_MODEL_NAME,
            content=[{"type": "text", "text": "Reply with the word ready."}],
            max_tokens=64,
            timeout=600,
            chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
        )
        if result.category == OK:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"no completion through the router: {result}")
        time.sleep(10)


def build_model_config(round_name: str) -> dict[str, Any]:
    return {
        "model": SERVED_MODEL_NAME,
        "test_name": f"muse-spark-pd-overload-{round_name}",
        "variant": "PD-1P1D-TP8-fp8KV-Raiden",
        "gates": [
            {**GSM8K_GATE, "label": f"gsm8k_{round_name}"},
            {**MMMU_PRO_GATE, "label": f"mmmu_pro_vision_{round_name}"},
        ],
    }


def run_quality_round(round_name: str, base_url: str) -> str | None:
    """Score both gates; return why the round failed, or None."""
    try:
        run_model_config(build_model_config(round_name), base_url=base_url)
    except Exception as exc:
        return f"{round_name} quality round failed: {exc}"
    return None


def run_overload_flow(
    *,
    router_url: str,
    engine_urls: Sequence[str],
    healthy_at: float,
    liveness_failures: Callable[[], list[str]],
) -> dict[str, Any]:
    """Overload with evals, stop the load, drain, evaluate again; report it all.

    `healthy_at` is when both engines answered /health (time.monotonic()); the
    flow limit counts from there.
    """
    started = time.monotonic()
    usage_ceiling = (
        max(scrape_engine(url).token_usage for url in engine_urls) + IDLE_USAGE_MARGIN
    )
    sizer = measure_prompt_sizer(
        router_url, model=SERVED_MODEL_NAME, timeout=REQUEST_TIMEOUT_S
    )
    relay = EvalRetryRelay(
        router_url,
        retry_budget_s=REQUEST_TIMEOUT_S,
        request_timeout_s=REQUEST_TIMEOUT_S,
    ).start()
    monitor = MetricsMonitor(engine_urls).start()
    driver = OverloadDriver(
        router_url,
        model=SERVED_MODEL_NAME,
        sizer=sizer,
        mix=LOAD_MIX,
        clients=LOAD_CLIENTS,
        probe_prompt_tokens=PROBE_PROMPT_TOKENS,
        max_tokens=LOAD_MAX_TOKENS,
        probe_max_tokens=PROBE_MAX_TOKENS,
        request_timeout_s=REQUEST_TIMEOUT_S,
        # A probe still shed at the flow limit has failed either way.
        probe_deadline_s=max(60.0, healthy_at + FLOW_LIMIT_S - time.monotonic()),
        burst_size=BURST_SIZE,
        burst_interval_s=BURST_INTERVAL_S,
        chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
    ).start()
    failures: list[str] = []
    try:
        time.sleep(OVERLOAD_WARMUP_S)
        # The overload has to hold while the first round scores, so its evidence
        # is counted from here: the clients' startup ramp draws 429s by itself.
        monitor.reset_peaks()
        shed_before_evals = driver.shed_count()
        rounds = {"under_load": run_quality_round("under_load", relay.url)}
        shed_during_evals = driver.shed_count() - shed_before_evals
        eval_429s_retried = relay.retried_429
        peak_queued_during_evals = dict(monitor.peak_queued)
        driver.stop_new_traffic()
        if not driver.join(DRAIN_TIMEOUT_S):
            failures.append(
                f"load still in flight {DRAIN_TIMEOUT_S}s after traffic stopped"
            )
        monitor.stop()
        drain = wait_until_idle(
            engine_urls, timeout=DRAIN_TIMEOUT_S, usage_ceiling=usage_ceiling
        )
        rounds["after_drain"] = run_quality_round("after_drain", relay.url)
    finally:
        driver.stop_new_traffic()
        monitor.stop()
        relay.stop()
    flow_s = time.monotonic() - started
    since_healthy_s = time.monotonic() - healthy_at

    failures += [failure for failure in rounds.values() if failure]
    failures += driver.failures()
    failures += [f"eval request: {failure}" for failure in relay.failures]
    if (
        shed_during_evals + eval_429s_retried == 0
        and max(peak_queued_during_evals.values()) == 0
    ):
        failures.append("no 429s and nothing queued while the evals ran")
    if not drain.idle:
        failures.append(
            f"engines not idle {drain.waited_s:.0f}s after the load ended: "
            f"{drain.loads} {drain.error or ''}"
        )
    failures += liveness_failures()
    if since_healthy_s > FLOW_LIMIT_S:
        failures.append(
            f"flow ended {since_healthy_s:.0f}s after the engines turned healthy, "
            f"over the {FLOW_LIMIT_S}s limit"
        )
    return {
        "flow_s": round(flow_s),
        "since_healthy_s": round(since_healthy_s),
        "sizer": {
            "tokens_per_line": sizer.tokens_per_line,
            "overhead_tokens": sizer.overhead_tokens,
        },
        "load": driver.summary(),
        "load_429s_during_evals": shed_during_evals,
        "eval_relay": {
            "answered": relay.answered,
            "retried_429_under_load": eval_429s_retried,
            "retried_429_total": relay.retried_429,
        },
        "peak_queued_during_evals": peak_queued_during_evals,
        "metrics_scrape_errors": monitor.errors,
        "drain": {"idle": drain.idle, "waited_s": round(drain.waited_s)},
        "failures": failures,
    }


class TestMuseSparkPdOverloadIntegration(PDSingleHostTestCase):
    """Run the stress flow on the prefill pod; the decode pod only serves."""

    config_module = LAUNCH_CONFIG_MODULE
    repo_root = REPO
    cwd = BENCH
    base_url = f"http://{LOCAL_HOST}:{ROUTER_PORT}"
    log_path = PREFILL_LOG
    decode_log_path = DECODE_LOG
    self_hostname = staticmethod(self_hostname)
    decode_host_env = DECODE_HOST_ENV
    tag = "pd-singlehost-muse-overload"
    health_timeout = 2 * WAIT_SECS + 300
    healthy_at: ClassVar[float] = 0.0
    _router: ClassVar[RecipeLauncher | None] = None

    @classmethod
    def setUpClass(cls) -> None:
        validate_model_source(MODEL_PATH)
        cls.role = os.environ.get("CI_PD_ROLE")
        if cls.role != PREFILL_ROLE:
            super().setUpClass()
            return
        # The engine alone, launched as the decode pod launches its own: the
        # inherited prefill path would start the group's mini load balancer,
        # which has no admission cap.
        cls._launcher = cls._start_launcher(
            cls.config_module + ":PREFILL_CONFIG",
            {**cls.runtime_envs, "SGLANG_LAUNCHER_RANK": "0"},
            log_path=cls.log_path,
        )
        for url in (cls.prefill_url(), cls.decode_url()):
            wait_for_health(
                url,
                cls.health_timeout,
                process=cls._launcher.process,
                endpoint="/health",
                tag=cls.tag,
            )
        cls.healthy_at = time.monotonic()
        cls.check_server_configuration()
        router = RecipeLauncher(
            command=router_command(
                f"http://{cls.self_hostname()}:{ENGINE_PORT}", cls.decode_url()
            ),
            cwd=str(cls.repo_root),
            log_path=ROUTER_LOG,
            start_new_session=True,
            tag=cls.tag + "-router",
        )
        cls.addClassCleanup(router.shutdown)
        router.start()
        cls._router = router
        wait_for_health(
            cls.base_url,
            HEALTH_TIMEOUT,
            process=router.process,
            endpoint="/health",
            tag=cls.tag,
        )
        wait_for_first_completion(cls.base_url, HEALTH_TIMEOUT, router.process)

    @classmethod
    def prefill_url(cls) -> str:
        return f"http://{LOCAL_HOST}:{ENGINE_PORT}"

    @classmethod
    def decode_url(cls) -> str:
        return f"http://{cls.decode_hostname()}:{ENGINE_PORT}"

    @classmethod
    def check_server_configuration(cls) -> None:
        failures = engine_configuration_failures(
            prefill_info=_get_json(cls.prefill_url() + "/server_info"),
            decode_info=_get_json(cls.decode_url() + "/server_info"),
        )
        if failures:
            raise RuntimeError("; ".join(failures))

    @classmethod
    def liveness_failures(cls) -> list[str]:
        failures = []
        for name, process in (
            ("prefill launcher", cls._launcher.process),
            ("router", cls._router.process),
        ):
            if process.poll() is not None:
                failures.append(f"{name} exited with code {process.returncode}")
        # /health answers 200 with an empty body; urlopen raises on any other
        # status (HTTPError is an OSError) or a broken reply.
        try:
            with urllib.request.urlopen(cls.decode_url() + "/health", timeout=30):
                pass
        except (OSError, http.client.HTTPException) as exc:
            failures.append(f"decode engine stopped answering /health: {exc}")
        return failures

    def test_overload_stress(self) -> None:
        if self.role == DECODE_ROLE:
            self.run_evaluations()
            return
        report = run_overload_flow(
            router_url=self.base_url,
            engine_urls=(self.prefill_url(), self.decode_url()),
            healthy_at=self.healthy_at,
            liveness_failures=self.liveness_failures,
        )
        print(f"[{self.tag}] " + json.dumps(report, indent=2), flush=True)
        self.assertEqual(report["failures"], [], "\n".join(report["failures"]))


def external_group() -> dict[str, Any]:
    """This file's group in the external nightly's slice matrix."""
    rows = json.loads(
        subprocess.check_output(
            [sys.executable, RENDER_EXTERNAL_JOBSET, "--list-groups", "--nightly"],
            cwd=REPO,
            text=True,
        )
    )
    rel_path = os.path.relpath(__file__, REPO)
    groups = [
        group for row in rows for group in row["groups"] if group["files"] == rel_path
    ]
    if len(groups) != 1:
        raise LookupError(f"{rel_path} is in {len(groups)} external groups, not one")
    return groups[0]


def render_jobset_yaml(
    group: Mapping[str, Any],
    image: str = "us-central1-docker.pkg.dev/sgl-project/tpu-ci/sglang:latest",
) -> str:
    """Render the external nightly's JobSet for this file's group."""
    return subprocess.check_output(
        [
            sys.executable,
            RENDER_EXTERNAL_JOBSET,
            "--image",
            image,
            "--group",
            json.dumps(group),
            # CI resolves the slice's nodepool from a repo secret.
            "--nodepool",
            "dry-run-nodepool",
        ],
        cwd=REPO,
        text=True,
    )


class TestMuseSparkPdOverloadConfiguration(unittest.TestCase):
    """Launch-contract tests that do not initialize TPU hardware."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.plans = {}
        for role in (PREFILL_ROLE, DECODE_ROLE):
            output = subprocess.check_output(
                [
                    sys.executable,
                    "-m",
                    "benchmark.tpu.launch_server",
                    "--config",
                    LAUNCH_CONFIG_MODULE + ":" + role.upper() + "_CONFIG",
                    "--runtime-envs",
                    json.dumps(
                        {"FORCE_SINGLE_HOST": "1", "SGLANG_HOST_IP": "127.0.0.1"}
                    ),
                    "--dry-run",
                ],
                cwd=REPO,
                text=True,
            )
            cls.plans[role] = json.loads(output)

    def test_each_engine_owns_one_host_and_starts_no_router(self) -> None:
        # The test fronts the pair with its capped router; a planned mini load
        # balancer would take the port and admit everything.
        for plan in self.plans.values():
            self.assertEqual(plan["server_args"]["tp_size"], TP_SIZE)
            self.assertEqual(plan["server_args"]["nnodes"], HOSTS_PER_ENGINE)
            self.assertIsNone(plan["router_command"])

    def test_engines_match_the_server_configuration_check(self) -> None:
        # check_server_configuration fails the run on any of these, so the
        # recipe and its constants here must agree.
        infos = {role: plan["server_args"] for role, plan in self.plans.items()}
        self.assertEqual(
            engine_configuration_failures(
                prefill_info=infos[PREFILL_ROLE], decode_info=infos[DECODE_ROLE]
            ),
            [],
        )
        for args in infos.values():
            self.assertEqual(args["model_path"], MODEL_PATH)
            self.assertEqual(args["served_model_name"], SERVED_MODEL_NAME)
            self.assertEqual(args["port"], ENGINE_PORT)
        self.assertEqual(
            infos[PREFILL_ROLE]["disaggregation_bootstrap_port"], BOOTSTRAP_PORT
        )

    def test_only_the_prefill_engine_offloads_kv(self) -> None:
        decode_args = self.plans[DECODE_ROLE]["server_args"]
        self.assertFalse(any(name.startswith("hicache") for name in decode_args))
        self.assertNotIn("enable_hierarchical_cache", decode_args)

    def test_router_sheds_with_429_instead_of_queueing(self) -> None:
        command = router_command("http://p:30000", "http://d:30000")
        self.assertNotIn("--mini-lb", command)
        flag = command.index("--queue-size")
        self.assertEqual(command[flag + 1], "0")
        self.assertIn("--disable-retries", command)
        self.assertIn("--disable-circuit-breaker", command)

    def test_each_burst_exceeds_a_full_router_bucket(self) -> None:
        # Steady closed-loop load is all admitted once the bucket's refill has
        # caught up; only a burst larger than a full bucket must draw 429s.
        self.assertGreater(BURST_SIZE, ROUTER_MAX_CONCURRENT)
        command = router_command("http://p:30000", "http://d:30000")
        flag = command.index("--max-concurrent-requests")
        self.assertEqual(int(command[flag + 1]), ROUTER_MAX_CONCURRENT)

    def test_longest_probe_fits_the_context(self) -> None:
        self.assertLessEqual(
            max(PROBE_PROMPT_TOKENS) + PROBE_MAX_TOKENS, CONTEXT_LENGTH
        )
        for plan in self.plans.values():
            self.assertEqual(plan["server_args"]["context_length"], CONTEXT_LENGTH)

    def test_peer_hostname_resolution(self) -> None:
        # An external pod's PD_SELF_HOST: <jobset>-<role>-<job>-<pod>.<jobset>.
        prefill = "0123456789ab-abcde-prefill-0-0.0123456789ab-abcde"
        self.assertEqual(
            peer_hostname(prefill, from_role=PREFILL_ROLE, to_role=DECODE_ROLE),
            "0123456789ab-abcde-decode-0-0.0123456789ab-abcde",
        )
        self.assertEqual(
            self_hostname({SELF_HOST_ENV: "retained-prefill.example"}),
            "retained-prefill.example",
        )

    @unittest.skipUnless(
        os.path.exists(RENDER_EXTERNAL_JOBSET),
        "renderer is not shipped in the runtime image",
    )
    def test_external_nightly_runs_one_host_per_role(self) -> None:
        group = external_group()
        self.assertEqual(group["mode"], "pd")
        self.assertEqual(group["prefill_hosts"], HOSTS_PER_ENGINE)
        self.assertEqual(group["decode_hosts"], HOSTS_PER_ENGINE)
        # Both roles share one slice, a single-host engine per pod.
        self.assertFalse(group["pair"])
        rel_path = os.path.relpath(__file__, REPO)
        self.assertEqual(group["cmd"], f"python3 {rel_path} integration")
        # Peer discovery swaps the role segment of the pods' hostnames.
        for role in (PREFILL_ROLE, DECODE_ROLE):
            self.assertNotIn(role, group["run_name"])
        yaml_content = render_jobset_yaml(group)
        self.assertIn("name: prefill", yaml_content)
        self.assertIn("name: decode", yaml_content)
        self.assertIn("completions: 1", yaml_content)
        self.assertIn("CI_PD_ROLE", yaml_content)
        self.assertIn(group["cmd"], yaml_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Muse Spark 1P1D overload stress")
    parser.add_argument(
        "mode",
        nargs="?",
        default="unittest",
        choices=["unittest", "integration", "integration_dry_run"],
    )
    args, extra_args = parser.parse_known_args()

    if args.mode == "integration_dry_run":
        print(render_jobset_yaml(external_group()))
        sys.exit(0)
    test_case = (
        TestMuseSparkPdOverloadIntegration
        if args.mode == "integration"
        else TestMuseSparkPdOverloadConfiguration
    )
    unittest.main(
        defaultTest=test_case.__name__,
        argv=[sys.argv[0], *extra_args],
        verbosity=2,
    )
