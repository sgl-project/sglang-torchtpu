"""GPT-OSS-120B EAGLE3 two-host GSM8K quality gate at high concurrency."""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST
from sglang.test.tpu import model_config_main

register_tpu_ci(
    est_time=3600,
    suite="external-nightly-2x2x2",
    nightly=True,
    labels=["multihost", "gpt-oss-120b", "spec"],
)

MULTIHOST_LAUNCH = {
    "launch_script": "recipe/gpt_oss_120b_spec_mh.sh",
    "weights": "gpt-oss-120b",
}

SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", DEFAULT_URL_FOR_TEST)
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "3000"))
GSM8K_ACCURACY = 0.70
GSM8K_CONCURRENCY = 64
SPEC_ACCEPT_LENGTH_THRESHOLD = float(
    os.environ.get("SPEC_ACCEPT_LENGTH_THRESHOLD", "1.90")
)

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-multihost-spec-gsm8k",
    "variant": "2host-TP16+EP16+DP4+EAGLE3-c64",
    "launch": {
        "external": True,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-multihost-spec-gsm8k-e2e",
    },
    "skip_unless": (
        os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1",
        "multi-host live-server not provided (the JobSet launches the server)",
    ),
    "gates": [
        {
            "dataset": "gsm8k",
            "label": "gsm8k",
            "baseline": GSM8K_ACCURACY,
            "spec_accept_length_threshold": SPEC_ACCEPT_LENGTH_THRESHOLD,
            "num_examples": 200,
            "num_threads": GSM8K_CONCURRENCY,
            "max_tokens": 512,
            "temperature": 0.0,
        }
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
