"""Extra multi-host checks that SHARE the Qwen3-30B-A3B-FP8 server.

Its MULTIHOST_LAUNCH is identical to test_tpu_30b_multihost.py, so the CI groups the two
files: one JobSet, one launch_server.sh bring-up (the expensive 30B compile happens
once), and rank-0 runs BOTH files against the same live server -- no second launch. This
file demonstrates that grouping and adds robust, model-agnostic serving checks.
"""

import os
import unittest

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, CustomTestCase

register_tpu_ci(est_time=300, suite="nightly-tpu-multihost", nightly=True)

# Identical to test_tpu_30b_multihost.py -> same launch group -> shared server.
MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/qwen3_30b_a3b_fp8_mh.sh",
    "weights_pvc": "model-weights-rox",
}

BASE_URL = os.environ.get("BASE_URL", DEFAULT_URL_FOR_TEST)


@unittest.skipUnless(
    os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1",
    "multi-host live-server not provided (the JobSet launches the server)",
)
class TestTpu30BExtra(CustomTestCase):
    """Robust, model-agnostic checks against the shared multi-host server."""

    def test_respects_max_new_tokens(self):
        resp = requests.post(
            f"{BASE_URL}/generate",
            json={
                "text": "Count up from one:",
                "sampling_params": {"max_new_tokens": 12, "temperature": 0},
            },
            timeout=180,
        )
        self.assertEqual(resp.status_code, 200)
        out = resp.json()
        self.assertTrue(out["text"])  # non-empty generation
        self.assertLessEqual(out["meta_info"]["completion_tokens"], 12)

    def test_batch_of_prompts(self):
        resp = requests.post(
            f"{BASE_URL}/generate",
            json={
                "text": ["Hello,", "The opposite of hot is"],
                "sampling_params": {"max_new_tokens": 8, "temperature": 0},
            },
            timeout=180,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()), 2)


if __name__ == "__main__":
    unittest.main()
