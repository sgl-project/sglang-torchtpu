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

register_tpu_ci(est_time=600, suite="stage-b-test-tpu-4-v7")

MODEL = "Qwen/Qwen3-0.6B"
EAGLE3_MODEL = "Qwen/Qwen3-1.7B"
EAGLE3_DRAFT_MODEL = "AngelSlim/Qwen3-1.7B_eagle3"


class TestTpuQwen306bSinglehostCompileTracer(CustomTestCase):
    def _run_server(
        self,
        tracer,
        *,
        model=MODEL,
        extra_args=(),
        expect_speculative_decoding=False,
    ):
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
        args.extend(extra_args)

        process = popen_launch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            device="tpu",
        )
        try:
            request_path = "/v1/completions"
            request_json = {
                "model": model,
                "prompt": "The capital of France is",
                "max_tokens": 8,
                "temperature": 0,
            }
            if expect_speculative_decoding:
                request_path = "/generate"
                request_json = {
                    "text": "The capital of France is",
                    "sampling_params": {"max_new_tokens": 16, "temperature": 0},
                }
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}{request_path}", json=request_json, timeout=120
            )
            self.assertEqual(response.status_code, 200)
            if expect_speculative_decoding:
                meta_info = response.json()["meta_info"]
                self.assertGreater(meta_info["spec_verify_ct"], 0)
                self.assertGreater(meta_info["spec_num_proposed_drafts"], 0)
                self.assertGreater(meta_info["spec_num_correct_drafts"], 0)

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

    def test_direct_with_eagle3(self):
        state = self._run_server(
            "direct",
            model=EAGLE3_MODEL,
            extra_args=(
                "--attention-backend",
                "pallas_tpu_rpa_v3",
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-draft-model-path",
                EAGLE3_DRAFT_MODEL,
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--disable-overlap-schedule",
                "--disable-radix-cache",
                "--context-length",
                "512",
                "--max-total-tokens",
                "512",
                "--max-prefill-tokens",
                "64",
                "--chunked-prefill-size",
                "64",
                "--tpu-compile-tokens",
                "64",
            ),
            expect_speculative_decoding=True,
        )
        self.assertEqual(state["torch_compile_dynamo_frames"], 0)


if __name__ == "__main__":
    unittest.main()
