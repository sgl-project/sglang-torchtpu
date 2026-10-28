"""GPT-OSS-120B P/D throughput on the experimental spec worker: a PP8 prefill
feeding a TP8/EP8 decode.

The JobSet gives each engine one full v7x host. The prefill runs eight
single-device pipeline stages and relays the drafter's aux hidden states to the
last stage, which hosts the draft; the decode pulls the KV and the draft state
from that stage over Raiden. The prefill pod runs the mini router and drives the
8k1k sweep through it. Subclasses pick the drafter's recipe.
"""

import json
import os
import socket
import subprocess
import sys
import unittest
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from sglang.test.test_utils import dump_metric
from sglang.test.tpu.utils.pd_singlehost_base import PDSingleHostTestCase

REPO = Path(__file__).absolute().parents[2]
BENCH = REPO / "benchmark" / "tpu"

SERVED_MODEL_NAME = "gpt-oss-120b"
DEFAULT_MODEL_PATH = "/weights/gpt-oss-120b"

PREFILL_ROLE = "prefill"
DECODE_ROLE = "decode"

# A v7x host has four physical chips and eight addressable TPU devices.
CHIPLETS_PER_HOST = 8
PREFILL_PP_SIZE = CHIPLETS_PER_HOST
DECODE_TP_SIZE = CHIPLETS_PER_HOST

PREFILL_PORT = int(os.environ.get("PREFILL_PORT", "30000"))
DECODE_PORT = int(os.environ.get("DECODE_PORT", "30000"))
LOCAL_HOST = "127.0.0.1"
WAIT_SECS = int(os.environ.get("WAIT_SECS", "5400"))

WORKLOAD = "8k1k"
ISL, OSL, CONCUR = 8192, 1024, 64

# Random-ids prompts give a drafter nothing to predict; pin acceptance near the
# GSM8K measurement, as the single-host spec perf tests do.
SIMULATE_ACC_LEN = 2.5
# The simulated acceptance is random per verify; a 640-request mean sits well inside this.
ACCEPT_LENGTH_TOLERANCE = 0.3
EXPERIMENTAL_SPEC_ENV = "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER"


def resolve_model_path(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    return values.get("GPT_OSS_MODEL") or values.get("MODEL") or DEFAULT_MODEL_PATH


def _self_hostname() -> str:
    if os.environ.get("PD_SELF_HOST"):
        return os.environ["PD_SELF_HOST"]
    hostnames = [h for h in os.environ.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    return hostnames[0] if hostnames else socket.gethostname()


def _get_json(url: str, timeout: int = 30) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def engine_configuration_failures(
    *,
    prefill_info: Mapping[str, Any],
    decode_info: Mapping[str, Any],
    algorithm: str,
) -> list[str]:
    expected = {
        PREFILL_ROLE: (prefill_info, {"tp_size": 1, "pp_size": PREFILL_PP_SIZE}),
        DECODE_ROLE: (decode_info, {"tp_size": DECODE_TP_SIZE, "pp_size": 1}),
    }
    failures = []
    for role, (info, sizes) in expected.items():
        if info.get("disaggregation_mode") != role:
            failures.append(
                f"{role} reports disaggregation_mode="
                f"{info.get('disaggregation_mode')!r}, expected {role!r}"
            )
        if str(info.get("speculative_algorithm", "")).upper() != algorithm:
            failures.append(
                f"{role} reports speculative_algorithm="
                f"{info.get('speculative_algorithm')!r}, expected {algorithm!r}"
            )
        for name, value in sizes.items():
            if info.get(name) != value:
                failures.append(
                    f"{role} reports {name}={info.get(name)!r}, expected {value}"
                )
    return failures


def read_spec_accept_length(metrics_text: str) -> float:
    """Max sglang:spec_accept_length across ranks; 0.0 when spec never ran."""
    values = [
        float(line.rsplit(" ", 1)[1])
        for line in metrics_text.splitlines()
        if line.startswith("sglang:spec_accept_length") and " " in line
    ]
    return max(values, default=0.0)


class PDPipelineSpecPerfTestCase(PDSingleHostTestCase):
    """Subclasses set config_module, algorithm, log paths, tag and the dashboard ids."""

    algorithm: ClassVar[str] = ""
    test_name: ClassVar[str] = ""
    metric_id: ClassVar[str] = ""
    display_name: ClassVar[str] = ""
    variant: ClassVar[str] = ""

    repo_root = REPO
    cwd = str(BENCH)
    self_hostname = staticmethod(_self_hostname)
    decode_host_env = "PD_DECODE_HOST"
    health_timeout = 2 * WAIT_SECS + 300
    health_generate_timeout = float(os.environ.get("HEALTH_TIMEOUT", "5400"))
    # Class-level so the decode role, which skips get_runtime_envs, gets it too.
    runtime_envs: ClassVar[dict[str, str | None]] = {
        "SGLANG_SIMULATE_ACC_LEN": str(SIMULATE_ACC_LEN)
    }

    @classmethod
    def build_model_config(cls) -> dict[str, Any]:
        return {
            "model": SERVED_MODEL_NAME,
            "test_name": cls.test_name,
            "id": cls.metric_id,
            "display_name": cls.display_name,
            "variant": cls.variant,
            "gates": [
                {
                    "dataset": "bench_serving",
                    "label": WORKLOAD,
                    "sweep": [
                        {
                            "workload": WORKLOAD,
                            "isl": ISL,
                            "osl": OSL,
                            "concurrency": CONCUR,
                        }
                    ],
                    "bench_env": {
                        "DATASET_NAME": "random-ids",
                        "TOKENIZE_PROMPT": "1",
                        "MODEL": resolve_model_path(),
                        "SERVED_MODEL_NAME": SERVED_MODEL_NAME,
                    },
                }
            ],
        }

    @classmethod
    def check_server_configuration(cls) -> None:
        failures = engine_configuration_failures(
            prefill_info=_get_json(f"http://{LOCAL_HOST}:{PREFILL_PORT}/server_info"),
            decode_info=_get_json(
                f"http://{cls.decode_hostname()}:{DECODE_PORT}/server_info"
            ),
            algorithm=cls.algorithm,
        )
        if failures:
            raise RuntimeError("; ".join(failures))

    def test_throughput(self) -> None:
        self.run_evaluations()
        if self.role != PREFILL_ROLE:
            return
        url = f"http://{self.decode_hostname()}:{DECODE_PORT}/metrics"
        with urllib.request.urlopen(url, timeout=60) as response:
            accept_length = read_spec_accept_length(response.read().decode("utf-8"))
        dump_metric(
            "spec_accept_length",
            accept_length,
            labels={
                "kind": "info",
                "id": self.metric_id,
                "display_name": self.display_name,
                "model": SERVED_MODEL_NAME,
                "workload": WORKLOAD,
            },
        )
        self.assertAlmostEqual(
            accept_length,
            SIMULATE_ACC_LEN,
            delta=ACCEPT_LENGTH_TOLERANCE,
            msg=f"decode accept length {accept_length} on {url}: {self.algorithm} "
            "or the simulated acceptance did not take effect",
        )


class PDPipelineSpecConfigurationTestCase(unittest.TestCase):
    """Launch-contract checks of a PP spec P/D recipe; no TPU needed."""

    config_module: ClassVar[str] = ""
    algorithm: ClassVar[str] = ""
    perf_case: ClassVar[type[PDPipelineSpecPerfTestCase]]
    # DP attention size of the decode; 1 runs it without DP attention.
    decode_dp: ClassVar[int] = 1
    # Role -> the launcher's dry-run plan.
    plans: ClassVar[dict[str, dict[str, Any]]]

    @classmethod
    def setUpClass(cls) -> None:
        if not cls.config_module:
            raise unittest.SkipTest("abstract configuration test")
        cls.plans = {
            role: json.loads(
                subprocess.check_output(
                    [
                        sys.executable,
                        "-m",
                        "benchmark.tpu.launch_server",
                        "--config",
                        f"{cls.config_module}:{role.upper()}_CONFIG",
                        "--runtime-envs",
                        json.dumps(
                            {"FORCE_SINGLE_HOST": "1", "SGLANG_HOST_IP": "127.0.0.1"}
                        ),
                        "--dry-run",
                    ],
                    cwd=REPO,
                    text=True,
                )
            )
            for role in (PREFILL_ROLE, DECODE_ROLE)
        }

    def test_decode_dp_attention_matches_the_config(self) -> None:
        args = self.plans[DECODE_ROLE]["server_args"]
        if self.decode_dp == 1:
            self.assertFalse(args.get("enable_dp_attention", False))
            return
        self.assertEqual(args["dp_size"], self.decode_dp)
        self.assertTrue(args["enable_dp_attention"])
        # The drafts share the target lm_head inside the attention-TP group.
        self.assertTrue(args["enable_dp_lm_head"])

    def test_prefill_is_eight_single_device_stages(self) -> None:
        args = self.plans[PREFILL_ROLE]["server_args"]
        self.assertEqual(args["pp_size"], PREFILL_PP_SIZE)
        self.assertEqual(args["tp_size"], 1)
        self.assertTrue(args["disable_overlap_schedule"])

    def test_decode_is_tp8_and_pulls_pipeline_pools(self) -> None:
        plan = self.plans[DECODE_ROLE]
        self.assertEqual(plan["server_args"]["tp_size"], DECODE_TP_SIZE)
        self.assertEqual(plan["server_args"]["ep_size"], DECODE_TP_SIZE)
        self.assertEqual(plan["server_args"].get("pp_size", 1), 1)
        self.assertIn("SGLANG_RAIDEN_PIPELINE_TRANSFER", plan["runtime_env_keys"])
        # The perf test reads the decode's accept length gauge.
        self.assertTrue(plan["server_args"]["enable_metrics"])

    def test_both_engines_serve_the_same_spec_on_the_experimental_worker(self) -> None:
        prefill, decode = (self.plans[r]["server_args"] for r in ("prefill", "decode"))
        spec = {k: v for k, v in prefill.items() if k.startswith("speculative_")}
        self.assertEqual(spec["speculative_algorithm"], self.algorithm)
        # The draft KV pool and seed cross the handoff, so their layouts must match.
        self.assertEqual(
            spec, {k: v for k, v in decode.items() if k.startswith("speculative_")}
        )
        for plan in self.plans.values():
            args = plan["server_args"]
            self.assertEqual(args["disaggregation_transfer_backend"], "raiden")
            self.assertEqual(args["max_running_requests"], CONCUR)
            self.assertIn(EXPERIMENTAL_SPEC_ENV, plan["runtime_env_keys"])

    def test_perf_config_builds_its_sweep_gate(self) -> None:
        from sglang.test.tpu.evals import model_config

        self.assertTrue(model_config.build_gates(self.perf_case.build_model_config()))

    def test_accept_length_reads_the_max_over_ranks(self) -> None:
        text = (
            "# TYPE sglang:spec_accept_length gauge\n"
            'sglang:spec_accept_length{tp_rank="0"} 2.4\n'
            'sglang:spec_accept_length{tp_rank="1"} 2.6\n'
            "sglang:spec_accept_rate 0.5\n"
        )
        self.assertEqual(read_spec_accept_length(text), 2.6)
        self.assertEqual(read_spec_accept_length("sglang:num_running_reqs 1\n"), 0.0)

    def test_engine_configuration_check(self) -> None:
        prefill = {
            "disaggregation_mode": PREFILL_ROLE,
            "speculative_algorithm": self.algorithm,
            "tp_size": 1,
            "pp_size": PREFILL_PP_SIZE,
        }
        decode = {
            "disaggregation_mode": DECODE_ROLE,
            "speculative_algorithm": self.algorithm,
            "tp_size": DECODE_TP_SIZE,
            "pp_size": 1,
        }
        check = lambda **kw: engine_configuration_failures(  # noqa: E731
            algorithm=self.algorithm, **kw
        )
        self.assertEqual(check(prefill_info=prefill, decode_info=decode), [])
        self.assertEqual(
            check(prefill_info={**prefill, "pp_size": 1}, decode_info=decode),
            ["prefill reports pp_size=1, expected 8"],
        )
