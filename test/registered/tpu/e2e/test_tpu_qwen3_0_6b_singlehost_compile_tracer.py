"""End-to-end tests for the TPU compilation tracers."""

import sys
import unittest
from copy import deepcopy
from pathlib import Path

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Support the CI runner's `python path/to/test.py` invocation as well as Bazel.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmark.tpu.recipe.configs.config_eval import server_args_to_cli
from benchmark.tpu.recipe.configs.qwen3_0_6b import CONFIG

register_tpu_ci(
    est_time=175,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "qwen3-0.6b", "compile-tracer"],
)

MODEL = "Qwen/Qwen3-0.6B"
EAGLE3_MODEL = "Qwen/Qwen3-1.7B"
EAGLE3_DRAFT_MODEL = "AngelSlim/Qwen3-1.7B_eagle3"


class TestTpuQwen306bSinglehostCompileTracer(CustomTestCase):
    def _run_server(
        self,
        tracer,
        *,
        model=MODEL,
        overrides=None,
        expect_speculative_decoding=False,
    ):
        config = deepcopy(CONFIG)
        if tracer == "direct":
            config["server_args"]["tpu_compile_tracer"] = "direct"
        config["server_args"].update(overrides or {})
        # The helper supplies the model and URL-derived connection flags.
        for name in ("model_path", "host", "port"):
            config["server_args"].pop(name, None)

        process = popen_launch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args_to_cli(config["server_args"]),
            env=config["runtime_env"],
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
            overrides={
                "attention_backend": "pallas_tpu_rpa_v3",
                "speculative_algorithm": "EAGLE3",
                "speculative_draft_model_path": EAGLE3_DRAFT_MODEL,
                "speculative_num_steps": 3,
                "speculative_eagle_topk": 1,
                "speculative_num_draft_tokens": 4,
                "disable_radix_cache": True,
                "context_length": 512,
                "max_total_tokens": 512,
                "max_prefill_tokens": 64,
                "chunked_prefill_size": 64,
                "tpu_compile_tokens": [64],
            },
            expect_speculative_decoding=True,
        )
        self.assertEqual(state["torch_compile_dynamo_frames"], 0)


if __name__ == "__main__":
    unittest.main()
