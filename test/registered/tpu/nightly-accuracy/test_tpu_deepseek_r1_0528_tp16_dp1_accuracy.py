"""Original DeepSeek-R1-0528 TP16/EP16 FP8-KV evaluation, no DP attention.

One pass each of GPQA, AIME25 and 250 MMLU questions at concurrency 128.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST
from sglang.test.tpu import model_config_main

register_tpu_ci(
    est_time=10800,
    suite="nightly-tpu-multihost-accuracy",
    nightly=True,
    labels=["multihost", "deepseek-r1", "accuracy"],
)

MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/configs/deepseek_r1_0528/accuracy.py",
    "weights_pvc": "model-weights",
    "timeout_per_file": 10800,
}

BASE_URL = os.environ.get("BASE_URL", DEFAULT_URL_FOR_TEST)
MODEL_CONFIG = {
    "id": "accuracy-deepseek-r1-0528",
    "display_name": "DeepSeek R1-0528",
    "dashboard": "accuracy",
    "dashboard_model": "deepseek-r1-0528",
    "model": os.environ.get("MODEL", "/weights/DeepSeek-R1-0528"),
    "test_name": "DeepSeek-R1-0528-multihost-fp8kv",
    "variant": "2host-TP16+EP16+DP1-FP8KV",
    "launch": {
        "external": True,
        "health_timeout": float(os.environ.get("HEALTH_TIMEOUT", "3000")),
        "tag": "r1-fp8kv-e2e",
    },
    "skip_unless": (
        os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1",
        "multi-host live-server not provided (the JobSet launches the server)",
    ),
    "gates": [
        {
            "dataset": "gpqa",
            "label": "gpqa_diamond",
            "baseline": 0.715,
            "atol": 0.10,
            "repeat": 1,
            "num_threads": 128,
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 32768,
        },
        {
            "dataset": "aime25",
            "label": "aime25",
            # Preserve the existing serving-regression threshold.
            "baseline": 0.875,
            "atol": 0.20,
            "repeat": 1,
            "num_threads": 128,
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 32768,
        },
        {
            "dataset": "mmlu",
            "label": "mmlu",
            "report_only": True,
            "repeat": 1,
            # The same seeded 250 rows every model and every night: sgl-eval's
            # mmlu pins the archive revision and samples with sample_seed=0.
            "num_examples": 250,
            "num_threads": 128,
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 16384,
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
