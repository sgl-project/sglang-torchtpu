"""Smoke test for single-host TPU serving.

Launches a server with Qwen3-0.6B on a single TPU chip and verifies basic inference.
"""

import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_tpu_ci(
    est_time=70,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "qwen3-0.6b", "smoke"],
)
register_tpu_ci(
    est_time=300,
    suite="external-nightly-1host",
    nightly=True,
    labels=["singlehost", "qwen3-0.6b", "smoke"],
)

MODEL = "Qwen/Qwen3-0.6B"


def _skip_if_no_tpu():
    try:
        import importlib.util

        if importlib.util.find_spec("torch_tpu") is None:
            return True
    except Exception:
        return True
    return False


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestTpuQwen306bSinglehostSmoke(CustomTestCase):
    """Launch a TPU server and verify completions + chat endpoints."""

    @classmethod
    def setUpClass(cls):
        # setdefault: mode B runs these suites with SGLANG_PLATFORM=tpu_oot.
        os.environ.setdefault("SGLANG_PLATFORM", "tpu")
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--served-model-name",
                MODEL,
                "--dtype",
                "bfloat16",
                "--device",
                "tpu",
                "--grammar-backend",
                "none",
                # torch.compile is opt-in on TPU; enabling it auto-disables the
                # tc_piecewise prefill graph, which is unimplemented on TPU
                # (TpuDeviceModule has no graph_pool_handle) and otherwise crashes.
                "--enable-torch-compile",
                # Small shapes keep the per-shape compile grid + warmup fast.
                "--tp-size",
                "1",
                "--page-size",
                "64",
                "--chunked-prefill-size",
                "512",
                "--context-length",
                "1024",
                # Every request here is a single sub-16-token prompt, so one
                # extend bucket and one decode shape cover them all. Without
                # these the derived grid is 4 token buckets x 7 batch sizes and
                # precompile alone costs ~130s of the file's ~169s.
                "--torch-compile-max-bs",
                "1",
                "--tpu-compile-bs",
                "1",
                "--tpu-compile-tokens",
                "64",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_health_endpoint(self):
        resp = requests.get(f"{self.base_url}/health")
        self.assertEqual(resp.status_code, 200)

    def test_model_info(self):
        resp = requests.get(f"{self.base_url}/model_info")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("model_path", data)

    def test_served_model_name(self):
        resp = requests.get(f"{self.base_url}/v1/models")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["data"][0]["id"], MODEL)

    def test_completion_coherent(self):
        resp = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": MODEL,
                "prompt": "The capital of France is",
                "max_tokens": 10,
                "temperature": 0,
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        text = data["choices"][0]["text"]
        self.assertIn("Paris", text)

    def test_completion_usage_fields(self):
        resp = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": MODEL,
                "prompt": "2+2=",
                "max_tokens": 3,
                "temperature": 0,
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        usage = data["usage"]
        self.assertGreater(usage["prompt_tokens"], 0)
        self.assertGreater(usage["completion_tokens"], 0)
        self.assertEqual(
            usage["total_tokens"],
            usage["prompt_tokens"] + usage["completion_tokens"],
        )

    def test_chat_completion(self):
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 10,
                "temperature": 0,
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["choices"][0]["message"]["role"], "assistant")
        self.assertGreater(len(data["choices"][0]["message"]["content"]), 0)

    def test_multiple_sequential_requests(self):
        for prompt in ["Hello", "World", "Test"]:
            resp = requests.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "max_tokens": 3,
                    "temperature": 0,
                },
                timeout=120,
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertGreater(len(data["choices"][0]["text"]), 0)


if __name__ == "__main__":
    unittest.main()
