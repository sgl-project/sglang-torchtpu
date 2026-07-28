"""gpt-oss-120b multi-host e2e lm-eval (live-server mode).

Its MULTIHOST_LAUNCH forms its own launch group: recipe/gpt_oss_120b_mh.sh on 2 hosts
(tp=16, EP=8), MXFP4 weights from the ReadOnlyMany Hyperdisk PVC gptoss120b-rox
(one-time: .github/tpu-ci/hydrate-gptoss120b.yaml, then pvc-gptoss120b-rox.yaml).
The CI runs this file on rank-0 against the server the JobSet brought up; skips
unless a live server is provided.

Unlike the 30B smoke files, this asserts real accuracy: one pass of lm-eval
gpqa_diamond_cot_zeroshot (full 198-question diamond set, reasoning_effort low)
against test/lm_eval_configs/gpt-oss-120b-TPU.yaml -- ground truth is the model
card's low-effort GPQA Diamond (67.1) with the agreed 3-point absolute tolerance.
⚠️ Idavidrein/gpqa is a gated HF dataset: the CI pod needs an HF_TOKEN (see the
config yaml's note).
"""

import os
import time
import unittest

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.kits.lm_eval_kit import LMEvalMixin
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, CustomTestCase

register_tpu_ci(est_time=3600, suite="nightly-tpu-multihost", nightly=True)

# Own launch group: gpt-oss-120b MXFP4 recipe + the gpt-oss ROX weights volume.
MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/gpt_oss_120b_mh.sh",
    "weights_pvc": "gptoss120b-rox",
}

BASE_URL = os.environ.get("BASE_URL", DEFAULT_URL_FOR_TEST)
HEALTH_TIMEOUT = int(os.environ.get("HEALTH_TIMEOUT", "600"))


@unittest.skipUnless(
    os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1",
    "multi-host live-server not provided (the JobSet launches the server)",
)
class TestTpuGptOss120BLMEval(LMEvalMixin, CustomTestCase):
    """lm-eval accuracy against the 2-host gpt-oss-120b server."""

    model = "openai/gpt-oss-120b"
    model_config_name = "test/lm_eval_configs/gpt-oss-120b-TPU.yaml"
    base_url = BASE_URL

    @classmethod
    def setUpClass(cls):
        # The JobSet's rank-0 already gated on /health_generate; this recheck keeps
        # the file runnable by hand against any live server (BASE_URL=... HEALTH_TIMEOUT=...).
        deadline = time.perf_counter() + HEALTH_TIMEOUT
        while time.perf_counter() < deadline:
            try:
                response = requests.get(cls.base_url + "/health", timeout=10)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(5)
        raise TimeoutError(
            f"{cls.base_url}/health did not return 200 within {HEALTH_TIMEOUT}s"
        )


if __name__ == "__main__":
    unittest.main()
