# SPDX-License-Identifier: Apache-2.0
"""Single-host-per-engine P/D GPT-OSS-120B serving-time XLA recompilation gate."""

import argparse
import json
import os
import socket
import subprocess
import sys
import unittest
import urllib.request
from collections.abc import Mapping
from typing import Any

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.recompilation_gate import (
    assert_no_serving_recompilation,
    recompilation_gate_runner,
)
from sglang.test.tpu.utils.pd_singlehost_base import PDSingleHostTestCase

register_tpu_ci(
    est_time=1800,
    suite="tpu-integration",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "pd", "compile"],
)

E = os.environ.get

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 4)))
BENCH = os.path.join(REPO, "benchmark", "tpu")

PREFILL_ROLE = "prefill"
DECODE_ROLE = "decode"
SELF_HOST_ENV = "PD_SELF_HOST"

# Each engine uses one eight-chiplet TPU host.
HOSTS_PER_ENGINE = 1
CHIPLETS_PER_HOST = 8
TP_SIZE = HOSTS_PER_ENGINE * CHIPLETS_PER_HOST
EP_SIZE = TP_SIZE

MODEL_ENV = "GPT_OSS_MODEL"
DEFAULT_MODEL_PATH = "/weights/gpt-oss-120b"
SERVED_MODEL_NAME = "gpt-oss-120b"

PREFILL_PORT = int(E("PREFILL_PORT", "30000"))
DECODE_PORT = int(E("DECODE_PORT", "30000"))
LOCAL_HOST = "127.0.0.1"

PYTHON = E("PYTHON", "python3")
RENDER_JOBSET = os.path.join(REPO, ".github", "tpu-ci", "render_jobset.py")

# Must keep generated Kubernetes pod names <= 63 characters.
RUN_NAME = "gpt-oss-120b-pd-sh-recompile"

PREFILL_LOG = "/tmp/pd_singlehost_recompile_prefill.log"
DECODE_LOG = "/tmp/pd_singlehost_recompile_decode.log"
LAUNCH_CONFIG_MODULE = "recipe/configs/gpt_oss_120b/pd_singlehost.py"

DISAGG_TRANSFER_BACKEND = "raiden"


def resolve_model_path(env: Mapping[str, str] | None = None) -> str:
    """Resolve the checkpoint from the environment, else the mounted default."""
    values = os.environ if env is None else env
    return values.get(MODEL_ENV) or values.get("MODEL") or DEFAULT_MODEL_PATH


def validate_model_source(model_path: str) -> None:
    """Fail clearly when CI points at an unprovisioned local checkpoint."""
    if not os.path.isabs(model_path):
        return
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"GPT-OSS-120B checkpoint incomplete or missing at {model_path}, "
            f"provision it on the weights volume or set {MODEL_ENV}"
        )


def self_hostname(env: Mapping[str, str] | None = None) -> str:
    """This pod's own DNS name."""
    values = os.environ if env is None else env
    if values.get(SELF_HOST_ENV):
        return values[SELF_HOST_ENV]
    hostnames = [h for h in values.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    return hostnames[0] if hostnames else socket.gethostname()


def _get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def transport_failures(*, prefill_info: dict[str, Any], decode_info: dict[str, Any]):
    """Reject a run where the two engines are not a raiden P/D pair."""
    failures = []
    for label, info, expected_mode in (
        (PREFILL_ROLE, prefill_info, PREFILL_ROLE),
        (DECODE_ROLE, decode_info, DECODE_ROLE),
    ):
        mode = info.get("disaggregation_mode")
        if mode != expected_mode:
            failures.append(
                f"{label} engine reports disaggregation_mode={mode!r}, expected "
                f"{expected_mode!r}: it is not serving a P/D role"
            )
        backend = info.get("disaggregation_transfer_backend")
        if backend != DISAGG_TRANSFER_BACKEND:
            failures.append(
                f"{label} engine reports disaggregation_transfer_backend="
                f"{backend!r}, expected {DISAGG_TRANSFER_BACKEND!r}"
            )
    return failures


def pd_transport_runner(*, prefill_url: str, decode_url: str):
    """Verify that both servers report the expected P/D roles and backend."""

    def run(base_url: str) -> dict[str, Any]:
        prefill_info = _get_json(f"{prefill_url}/server_info")
        decode_info = _get_json(f"{decode_url}/server_info")
        failures = transport_failures(
            prefill_info=prefill_info, decode_info=decode_info
        )
        print(
            f"[pd-singlehost-recompile] lb={base_url} prefill="
            f"{prefill_info.get('disaggregation_mode')}/"
            f"{prefill_info.get('disaggregation_transfer_backend')} decode="
            f"{decode_info.get('disaggregation_mode')}/"
            f"{decode_info.get('disaggregation_transfer_backend')}",
            flush=True,
        )
        return {
            "score": 0.0 if failures else 1.0,
            "_passed": not failures,
            "_failures": failures,
        }

    return run


def build_model_config(*, decode_host: str) -> dict[str, Any]:
    """Assemble the P/D transport and multi-shape recompilation gates."""
    prefill_url = f"http://{LOCAL_HOST}:{PREFILL_PORT}"
    decode_url = f"http://{decode_host}:{DECODE_PORT}"
    return {
        "model": SERVED_MODEL_NAME,
        "test_name": RUN_NAME,
        "variant": f"PD-singlehost-TP{TP_SIZE}+EP{EP_SIZE}-raiden-recompile",
        "gates": [
            {
                "dataset": "pd_transport",
                "label": "pd-transport",
                "runner": pd_transport_runner(
                    prefill_url=prefill_url,
                    decode_url=decode_url,
                ),
            },
            {
                "dataset": "recompilation_multi_shape",
                "label": "pd-recompilation-multi-shape",
                "runner": recompilation_gate_runner(
                    engine_urls=[prefill_url, decode_url],
                    run_workload=True,
                    prompt_word_counts=(8, 96, 320),
                    batch_sizes=(1, 2, 4),
                    max_new_tokens=16,
                    use_baseline_diff=True,
                ),
            },
        ],
    }


class TestPdSingleHostGptOss120bRecompileIntegration(PDSingleHostTestCase):
    """Run multi-shape serving requests and assert zero XLA recompilations on P and D."""

    config_module = LAUNCH_CONFIG_MODULE
    repo_root = REPO
    cwd = BENCH
    log_path = PREFILL_LOG
    decode_log_path = DECODE_LOG
    self_hostname = staticmethod(self_hostname)
    runtime_envs = {
        "SGLANG_IS_IN_CI": None,
        "SGLANG_TPU_HOST_PADDING": "1",
        "SGLANG_TPU_CHECK_RECOMPILATION": "1",
        "SGLANG_TPU_CHECK_RECOMPILATION_SITES": "1",
    }

    @classmethod
    def build_model_config(cls) -> dict[str, Any]:
        return build_model_config(decode_host=cls.decode_hostname())

    @classmethod
    def setUpClass(cls) -> None:
        validate_model_source(resolve_model_path())
        super().setUpClass()

    def test_no_serving_recompilation(self) -> None:
        self.run_evaluations()
        if self.role == PREFILL_ROLE:
            assert_no_serving_recompilation(
                [
                    f"http://{LOCAL_HOST}:{PREFILL_PORT}",
                    f"http://{self.decode_hostname()}:{DECODE_PORT}",
                ]
            )


def render_jobset_yaml(
    image: str = "us-central1-docker.pkg.dev/sgl-project/tpu-ci/sglang:latest",
    lane: str = "nightly",
    weights_pvc: str = "model-weights",
    run_name: str = RUN_NAME,
) -> str:
    """Render the JobSet for this lane: one host of prefill, one of decode."""
    rel_path = os.path.relpath(__file__, REPO)
    cmd = [
        PYTHON,
        RENDER_JOBSET,
        "--pd-prefill-hosts",
        str(HOSTS_PER_ENGINE),
        "--pd-decode-hosts",
        str(HOSTS_PER_ENGINE),
        "--lane",
        lane,
        "--image",
        image,
        "--weights-pvc",
        weights_pvc,
        "--name",
        run_name,
        "--files",
        rel_path,
        "--exec",
        f"python3 {rel_path} integration",
    ]
    return subprocess.check_output(cmd, text=True)


class TestPdSingleHostGptOss120bRecompileConfiguration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

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
                        {
                            "FORCE_SINGLE_HOST": "1",
                            "SGLANG_HOST_IP": "127.0.0.1",
                            "SGLANG_IS_IN_CI": None,
                            "SGLANG_TPU_HOST_PADDING": "1",
                            "SGLANG_TPU_CHECK_RECOMPILATION": "1",
                            "SGLANG_TPU_CHECK_RECOMPILATION_SITES": "1",
                        }
                    ),
                    "--dry-run",
                ],
                cwd=REPO,
                text=True,
            )
            cls.plans[role] = json.loads(output)

    def test_each_engine_takes_one_whole_host(self) -> None:
        for plan in self.plans.values():
            self.assertEqual(plan["server_args"]["tp_size"], CHIPLETS_PER_HOST)
            self.assertEqual(plan["server_args"]["ep_size"], CHIPLETS_PER_HOST)

    def test_run_name_fits_the_generated_pod_name(self) -> None:
        pod_name = f"tpu-ci-{RUN_NAME}-xxxxx-prefill-0-0-xxxxx"
        self.assertLessEqual(len(pod_name), 63, pod_name)

    def test_model_config_wires_transport_and_recompilation_gates(self) -> None:
        cfg = build_model_config(decode_host="decode-0")
        labels = [g["label"] for g in cfg["gates"]]
        self.assertEqual(labels, ["pd-transport", "pd-recompilation-multi-shape"])

    @unittest.skipUnless(
        os.path.exists(RENDER_JOBSET), "renderer is not shipped in the runtime image"
    )
    def test_dry_run_render_jobset_yaml(self) -> None:
        yaml_content = render_jobset_yaml()
        self.assertIn(f"generateName: tpu-ci-{RUN_NAME}-", yaml_content)
        self.assertIn("name: prefill", yaml_content)
        self.assertIn("name: decode", yaml_content)
        self.assertIn(
            "python3 test/registered/tpu/e2e/test_tpu_gpt_oss_120b_pd_singlehost_recompile.py integration",
            yaml_content,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-host-per-engine P/D recompilation gate"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="unittest",
        choices=["unittest", "integration", "integration_dry_run"],
        help="Execution mode (default: unittest)",
    )
    args, extra_args = parser.parse_known_args()

    if args.mode == "integration_dry_run":
        print(render_jobset_yaml())
        sys.exit(0)

    test_case = (
        TestPdSingleHostGptOss120bRecompileIntegration
        if args.mode == "integration"
        else TestPdSingleHostGptOss120bRecompileConfiguration
    )
    unittest.main(
        defaultTest=test_case.__name__,
        argv=[sys.argv[0], *extra_args],
        verbosity=2,
    )
