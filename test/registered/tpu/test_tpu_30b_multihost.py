"""Qwen3-30B-A3B (FP8) multi-host serving smoke (live-server mode).

Its MULTIHOST_LAUNCH differs from the 0.6B files (different launch script + weights), so
the CI puts it in its own launch group -> its own JobSet + server, tp=16 / EP=8, weights
from the ReadOnlyMany Hyperdisk at /weights. Live-server: connect only. The CI runs this
file against the server rank-0 brought up; skips unless a live server is provided.
"""

import os
import unittest

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, CustomTestCase

register_tpu_ci(est_time=1800, suite="nightly-tpu-multihost", nightly=True)

# Own launch group: 30B-A3B-FP8 MoE recipe + the ROX weights volume.
MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/qwen3_30b_a3b_fp8_mh.sh",
    "weights_pvc": "qwen30b-rox",
}

BASE_URL = os.environ.get("BASE_URL", DEFAULT_URL_FOR_TEST)


@unittest.skipUnless(
    os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1",
    "multi-host live-server not provided (the JobSet launches the server)",
)
class TestTpu30BMultihost(CustomTestCase):
    """The tp=16 Qwen3-30B-A3B-FP8 server serves coherent, correct inference."""

    def _generate(self, text, max_new_tokens=16):
        resp = requests.post(
            f"{BASE_URL}/generate",
            json={
                "text": text,
                "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
            },
            timeout=180,
        )
        self.assertEqual(resp.status_code, 200)
        return resp.json()["text"]

    def test_health(self):
        resp = requests.get(f"{BASE_URL}/health_generate", timeout=60)
        self.assertEqual(resp.status_code, 200)

    def test_factual_inference(self):
        self.assertIn("Paris", self._generate("The capital of France is"))

    def test_arithmetic(self):
        # A cheap coherence check that the 30B MoE routing produced sane output.
        self.assertIn("4", self._generate("2 + 2 = "))


if __name__ == "__main__":
    unittest.main()
