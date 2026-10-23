"""Kimi-K3 Raiden P/D on a tiny random-weight model: one TP4 host per engine.

Kimi-K3 interleaves MLA layers, whose latent KV lives in the paged fused MLA
pool, with KDA layers, whose conv and recurrent state lives in a slot-indexed
Mamba pool. A disaggregated request must carry both to the decode engine. The
prefill pod first serves the requests colocated, then as the prefill engine
behind a mini-LB router; greedy output must match token for token. A decode
engine that resumed from a stale or zero KDA state, or from wrong MLA pages,
continues differently, so the comparison checks both transfers.

Every pod builds the same seeded checkpoint (kimi_k3_tiny_harness.py).

The decode pod (``CI_PD_ROLE=decode``) only serves; the JobSet ends with the
prefill pod.
"""

import contextlib
import os
import socket
import subprocess
import sys
import tempfile
import unittest

from kimi_k3_tiny_harness import (
    PROMPTS,
    assert_no_serving_recompile,
    build_model,
    launch,
    read_log,
    run_requests,
    token_ranks,
)

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, CustomTestCase
from sglang.utils import wait_for_http_ready

register_tpu_ci(
    est_time=900,
    suite="tpu-integration",
    nightly=True,
    labels=["pd", "singlehost", "kimi-k3", "multimodal"],
)

PREFILL_PORT = 21000
DECODE_PORT = 21000
BOOTSTRAP_PORT = 21100
LB_PORT = 21200


def _self_hostname() -> str:
    """This pod's DNS name. JobSet pods resolve each other only by the
    subdomain-qualified name GKE puts in TPU_WORKER_HOSTNAMES."""
    hostnames = [h for h in os.environ.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    return hostnames[0] if hostnames else socket.gethostname()


def _decode_hostname() -> str:
    """The decode pod's DNS name: this pod's with its role segment swapped.

    (pd_singlehost_base has the same helper, but importing it would bring up
    the TPU runtime in this process and leave the servers no chips.)
    """
    hostname = _self_hostname()
    if "-prefill-" not in hostname:
        raise ValueError(f"hostname {hostname!r} has no '-prefill-' segment")
    return hostname.replace("-prefill-", "-decode-")


def _disagg_args(role: str) -> list[str]:
    args = [
        "--disaggregation-mode",
        role,
        "--disaggregation-transfer-backend",
        "raiden",
        # A disaggregated engine's warmup request needs its peer and the router.
        "--skip-server-warmup",
    ]
    if role == "prefill":
        args += ["--disaggregation-bootstrap-port", str(BOOTSTRAP_PORT)]
    else:
        args += ["--disaggregation-decode-retraction-backup", "cpu_tensor"]
    return args


@contextlib.contextmanager
def _serving(model, port, args, log_path, env=None):
    process, log = launch(model, port, args, log_path, env)
    try:
        yield process
    finally:
        kill_process_tree(process.pid, wait_timeout=60)
        log.close()


class TinyPDParity:
    """The P/D-vs-colocated comparison; a variant sets the decode's extra
    server arguments and environment, and may serve the requests its own way
    (a mixin, so importing it adds no test)."""

    decode_args: list[str] = []
    decode_env: dict[str, str] = {}

    def serve_requests(self, base_url: str, model: str) -> dict:
        """The requests the disaggregated deployment serves."""
        return run_requests(base_url, model)

    def test_disaggregated_output_matches_colocated(self):
        role = os.environ.get("CI_PD_ROLE")
        if role not in ("prefill", "decode"):
            self.skipTest("needs CI_PD_ROLE=prefill or decode (a P/D JobSet)")
        model_dir = tempfile.TemporaryDirectory(prefix="kimi_k3_tiny_")
        self.addCleanup(model_dir.cleanup)
        build_model(model_dir.name)
        model = model_dir.name
        logs = tempfile.mkdtemp(prefix="kimi_k3_pd_logs_")

        if role == "decode":
            with _serving(
                model,
                DECODE_PORT,
                _disagg_args("decode") + self.decode_args,
                f"{logs}/decode.log",
                env=self.decode_env,
            ) as process:
                # Serve until the prefill pod finishes and the JobSet ends.
                process.wait()
            return

        decode_host = _decode_hostname()
        prefill_log = f"{logs}/prefill.log"
        with _serving(model, PREFILL_PORT, _disagg_args("prefill"), prefill_log):
            prefill_url = f"http://{_self_hostname()}:{PREFILL_PORT}"
            decode_url = f"http://{decode_host}:{DECODE_PORT}"
            wait_for_http_ready(
                f"{decode_url}/health", DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
            )
            router = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "sglang_router.launch_router",
                    "--pd-disaggregation",
                    "--mini-lb",
                    "--prefill",
                    prefill_url,
                    str(BOOTSTRAP_PORT),
                    "--decode",
                    decode_url,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(LB_PORT),
                ]
            )
            try:
                lb_url = f"http://127.0.0.1:{LB_PORT}"
                wait_for_http_ready(f"{lb_url}/health", 300)
                disaggregated = self.serve_requests(lb_url, model)
            finally:
                kill_process_tree(router.pid, wait_timeout=30)
        assert_no_serving_recompile(self, read_log(prefill_log), "prefill")

        # The colocated reference's own greedy output.
        with _serving(model, PREFILL_PORT, [], f"{logs}/colocated.log"):
            expected = run_requests(f"http://127.0.0.1:{PREFILL_PORT}", model)
        # The rank of each disaggregated token in the reference's distribution
        # at that position, and, for scale, of another prompt's continuation.
        # (A second server: prompt logprobs change the compiled program.)
        with _serving(
            model,
            PREFILL_PORT,
            [],
            f"{logs}/reference_logprobs.log",
            env={"SGLANG_TPU_ENABLE_PROMPT_LOGPROB": "1"},
        ):
            reference = f"http://127.0.0.1:{PREFILL_PORT}"
            ranks = [
                token_ranks(reference, prompt, tokens)
                for prompt, tokens in zip(PROMPTS, disaggregated["text"])
            ]
            unrelated = [
                token_ranks(reference, prompt, expected["text"][(i + 1) % len(PROMPTS)])
                for i, prompt in enumerate(PROMPTS)
            ]
        self.assertGreater(expected["image"][1], 0)
        for i, (got, want) in enumerate(zip(disaggregated["text"], expected["text"])):
            same = next(
                (k for k, (a, b) in enumerate(zip(got, want)) if a != b), len(want)
            )
            print(
                f"request {i}: first {same} of {len(want)} tokens match; "
                f"ranks {ranks[i]}; unrelated continuation ranks {unrelated[i]}"
            )
        self.check_outputs(disaggregated, expected, ranks)

    def check_outputs(self, disaggregated, expected, ranks) -> None:
        """Equal attention TP: the decode runs the colocated computation."""
        self.assertEqual(disaggregated, expected)


class TestKimiK3TinyPD(TinyPDParity, CustomTestCase):
    """Equal attention TP: the decode pulls the MLA pages, and the KDA state
    row on raiden's state engine."""


if __name__ == "__main__":
    unittest.main()
