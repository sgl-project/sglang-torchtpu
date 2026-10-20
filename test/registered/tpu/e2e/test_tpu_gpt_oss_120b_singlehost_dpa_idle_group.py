"""GPT-OSS-120B single-host DPA DP2/attention-TP4 gate at one request per group.

Attention DP pads every decode to the attention-TP width, four rows above the
request cap, and a group idles beside the other whenever its request finishes
first.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

register_tpu_ci(
    est_time=3600,
    suite="external-nightly-1host",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "dpa", "hybrid"],
)

SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

GSM8K_ACCURACY = 0.70
DP_SIZE = 2

LAUNCH_ENV = {
    "MODEL": "/weights/gpt-oss-120b",
    "ISL": "4096",
    "OSL": "512",
    # One request per DP group.
    "CONCUR": str(DP_SIZE),
    "CONTEXT_LEN": "8192",
    "TPU_COMPILE_TOKENS": "512 1024 2048 4096",
    "MAX_TOTAL_TOKENS": "8192",
    "ENABLE_PROFILER_ENDPOINT": "0",
    # TP8 / DP2 gives two replicas whose attention is sharded TP4.
    "DP_SIZE": str(DP_SIZE),
    "EXTRA_ARGS": (
        "--reasoning-parser gpt-oss "
        f"--served-model-name {SERVED_MODEL_NAME} "
        "--tpu-compile-tracer direct"
    ),
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-dpa-idle-group-gsm8k",
    "variant": "TP8+EP8+DP2(attn-tp4)",
    "launch": {
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-dpa-idle-group-gsm8k-e2e",
    },
    "gates": [
        {
            "dataset": "gsm8k",
            "label": "gsm8k",
            "baseline": GSM8K_ACCURACY,
            "num_examples": 200,
            "num_threads": DP_SIZE,
            "max_tokens": 512,
            "temperature": 0.0,
        }
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
