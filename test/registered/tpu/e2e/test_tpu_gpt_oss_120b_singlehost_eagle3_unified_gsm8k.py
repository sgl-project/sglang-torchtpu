"""GPT-OSS-120B single-host EAGLE3 on the unified Full/SWA KV pool: full GSM8K score.

The server is the nightly backbone GSM8K config plus the draft and the unified
pool, so a drop against the backbone's 0.946 points at the unified pool or the
draft rather than at the eval.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

register_tpu_ci(
    est_time=3600,
    suite="external-nightly-1host",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "spec", "eagle3", "unified-kv"],
)

SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))
DRAFT_MODEL_PATH = resolve_tpu_model_path("lmsys/EAGLE3-gpt-oss-120b-bf16")

# Backbone nightly measures 0.946 on the same sampling; greedy speculative
# decoding should not move it past the tolerance.
GSM8K_ACCURACY = 0.946
GSM8K_ATOL = 0.05
SPEC_ACCEPT_LENGTH_THRESHOLD = float(
    os.environ.get("SPEC_ACCEPT_LENGTH_THRESHOLD", "1.90")
)

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-eagle3-unified-gsm8k",
    "variant": "TP8+EP8+EAGLE3+UnifiedKV",
    "launch": {
        "config": "recipe/configs/gpt_oss_120b/accuracy_gsm8k_unified_eagle3.py",
        "env": {"DRAFT_MODEL": DRAFT_MODEL_PATH},
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-120b-eagle3-unified-gsm8k-e2e",
    },
    # No `dashboard`: a feature config, kept off the backbone accuracy series.
    "gates": [
        {
            "dataset": "gsm8k",
            "label": "gsm8k_full",
            "baseline": GSM8K_ACCURACY,
            "atol": GSM8K_ATOL,
            "spec_accept_length_threshold": SPEC_ACCEPT_LENGTH_THRESHOLD,
            "num_examples": 1319,
            "num_threads": 32,
            "max_tokens": 4096,
            "temperature": 0.0,
            "reasoning_effort": "low",
        },
    ],
}

if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
