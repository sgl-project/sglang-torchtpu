"""Muse Spark single-host DFlash on the unified Full/SWA KV pool: full GSM8K score.

The server is the nightly backbone accuracy config plus the DFlash draft and the
unified pool, so a drop against the backbone's 0.9765 points at the unified pool or
the draft rather than at the eval.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import MUSE_SPARK, model_config_main

register_tpu_ci(
    est_time=3600,
    suite="external-nightly-1host",
    nightly=True,
    labels=[
        "singlehost",
        "muse-spark-open-816b-a42b",
        "spec",
        "dflash",
        "unified-kv",
    ],
)

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "5400"))

# Backbone nightly (test_tpu_muse_spark_gsm8k.py) measures 0.9765 on the same sampling.
GSM8K_ACCURACY = 0.9765
GSM8K_ATOL = 0.05
# The separate-pool kv-offload twin measured a 4.80 mean on 200 GSM8K questions.
SPEC_ACCEPT_LENGTH_THRESHOLD = float(
    os.environ.get("SPEC_ACCEPT_LENGTH_THRESHOLD", "4.0")
)

MODEL_CONFIG = {
    "model": MUSE_SPARK["served_model_name"],
    "test_name": "muse-spark-dflash-unified-gsm8k",
    "variant": "TP8+DFlash+UnifiedKV",
    "launch": {
        "config": "recipe/configs/muse_spark/accuracy_gsm8k_unified_dflash.py",
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "muse-spark-dflash-unified-gsm8k-e2e",
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
