"""Gemma-4-31B-it single-host GSM8K gate (bf16, TP8).

Covers the hybrid KV pools (50 sliding layers at head_dim 256, 10 global at 512)
and, through the server's image warmup, the eager vision path around the
compiled text model.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

register_tpu_ci(
    est_time=3600,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "gemma4-31b", "multimodal"],
)

SERVED_MODEL_NAME = "gemma4-31b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
# Budget: weight load plus the EXTEND/decode captures for both input variants.
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "3600"))

# Floor, not a tuned target: measured 0.98 over 200 examples.
GSM8K_ACCURACY = 0.85

LAUNCH_ENV = {
    "MODEL": "/weights/gemma-4-31B-it",
    # reconcile_tpu_batch_limits raises torch_compile_max_bs back to
    # max_running_requests, so keep them equal. 16 gives bs [1, 2, 4, 8, 12, 16].
    "CONC": "16",
    "TORCH_COMPILE_MAX_BS": "16",
    # Three token buckets instead of the default eight; 8192 must stay to cover
    # MAX_PREFILL_TOKENS.
    "TPU_COMPILE_TOKENS": "512 2048 8192",
    "ENABLE_PROFILER_ENDPOINT": "0",
    "EXTRA_ARGS": f"--served-model-name {SERVED_MODEL_NAME}",
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gemma4-31b-gsm8k",
    "variant": "TP8",
    "launch": {
        "recipe": "recipe/gemma4_31b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gemma4-31b-e2e",
    },
    "gates": [
        {
            "dataset": "gsm8k",
            "label": "gsm8k",
            "baseline": GSM8K_ACCURACY,
            "atol": 0.0,
            "num_examples": 200,
            "num_threads": 16,
            "max_tokens": 512,
            "temperature": 0.0,
        }
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
