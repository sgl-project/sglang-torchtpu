"""GPT-OSS-120B full data-parallel-attention single-host quality gate."""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

register_tpu_ci(est_time=3600, suite="nightly-tpu-4-v7", nightly=True)

SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

GSM8K_ACCURACY = 0.70
DP_SIZE = 8

LAUNCH_ENV = {
    "MODEL": "/weights/gpt-oss-120b",
    "ISL": "4096",
    "OSL": "512",
    "CONCUR": "16",
    "CONTEXT_LEN": "8192",
    "TPU_COMPILE_TOKENS": "512 1024 2048 4096",
    # Two requests per DP rank at the configured global concurrency.
    "TPU_COMPILE_BS": "1 2",
    "MAX_TOTAL_TOKENS": "8192",
    "ENABLE_PROFILER_ENDPOINT": "0",
    # TP8 / DP8 gives attention TP=1: this is full DPA, not hybrid TP+DPA.
    "DP_SIZE": str(DP_SIZE),
    "EXTRA_ARGS": (
        "--reasoning-parser gpt-oss "
        f"--served-model-name {SERVED_MODEL_NAME} "
        "--tpu-compile-tracer dynamo"
    ),
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-dpa-gsm8k",
    "variant": "TP8+EP8+DP8(attn-tp1)",
    "launch": {
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-dpa-gsm8k-e2e",
    },
    "gates": [
        {
            "dataset": "gsm8k",
            "label": "gsm8k",
            "baseline": GSM8K_ACCURACY,
            "num_examples": 200,
            "num_threads": 16,
            "max_tokens": 512,
            "temperature": 0.0,
        }
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
