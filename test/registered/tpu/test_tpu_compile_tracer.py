"""End-to-end tests for the TPU compilation tracers."""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_tpu_ci(est_time=300, suite="stage-b-test-tpu-4-v7")

MODEL = "Qwen/Qwen3-0.6B"


class TestTpuCompileTracer(CustomTestCase):
    def _run_server(self, tracer):
        args = [
            "--dtype",
            "bfloat16",
            "--device",
            "tpu",
            "--grammar-backend",
            "none",
            "--enable-torch-compile",
            "--tp-size",
            "1",
            "--page-size",
            "64",
            "--chunked-prefill-size",
            "64",
            "--context-length",
            "128",
            "--max-prefill-tokens",
            "64",
            "--max-total-tokens",
            "128",
            "--max-running-requests",
            "1",
            "--torch-compile-max-bs",
            "1",
            "--tpu-compile-tokens",
            "16",
            "--tpu-compile-bs",
            "1",
        ]
        if tracer == "direct":
            args.extend(("--tpu-compile-tracer", "direct"))

        process = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            device="tpu",
        )
        try:
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/v1/completions",
                json={
                    "model": MODEL,
                    "prompt": "The capital of France is",
                    "max_tokens": 8,
                    "temperature": 0,
                },
                timeout=120,
            )
            self.assertEqual(response.status_code, 200)

            server_info = requests.get(
                f"{DEFAULT_URL_FOR_TEST}/server_info", timeout=30
            ).json()
            state = server_info["internal_states"][0]
            self.assertEqual(state["tpu_compile_tracer"], tracer)
            return state
        finally:
            kill_process_tree(process.pid, wait_timeout=30)

    def test_dynamo_is_default(self):
        self._run_server("dynamo")

    def test_direct_bypasses_dynamo_and_aot_autograd(self):
        state = self._run_server("direct")
        self.assertEqual(state["torch_compile_dynamo_frames"], 0)
        self.assertEqual(state["torch_compile_aot_autograd_graphs"], 0)


if __name__ == "__main__":
    unittest.main()
