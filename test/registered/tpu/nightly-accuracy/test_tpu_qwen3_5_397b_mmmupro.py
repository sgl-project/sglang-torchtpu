"""Qwen3.5-397B-A17B TPU MMMU-Pro nightly score, all 1,730 standard-10 questions.

The JobSet runs this file's recipe on both hosts and rank 0 scores against it;
sgl-eval's own loader and grader, thinking on, with the model card's sampling.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST
from sglang.test.tpu import QWEN3_5_397B, make_runner, model_config_main

register_tpu_ci(
    est_time=9000,
    suite="nightly-tpu-multihost-accuracy",
    nightly=True,
    labels=["multihost", "qwen3.5-397b", "accuracy"],
)

CONFIG_MODULE = QWEN3_5_397B["config_module"]
MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/configs/qwen3_5_397b/server.py",
    "weights_pvc": "model-weights",
    # Which accuracy-test-tpu.yml lane runs this file.
    "schedule": "mmmu-gemma4-qwen35",
    # Measured 2 h 16 min for the full set at 128 concurrent requests.
    "timeout_per_file": 10800,
}

BASE_URL = os.environ.get("BASE_URL", DEFAULT_URL_FOR_TEST)


def build_model_config():
    """Assemble this file's gates against the model's identity."""
    return {
        # The dashboard's series join key: never renamed, never shown.
        "id": "accuracy-qwen3-5-397b-mmmupro",
        "display_name": QWEN3_5_397B["display_name"],
        "dashboard": "accuracy",
        "dashboard_model": "qwen3.5-397b",
        "model": QWEN3_5_397B["served_model_name"],
        "test_name": "Qwen3.5-397B-multihost-mmmupro",
        "variant": QWEN3_5_397B["variant"],
        "launch": {
            "external": True,
            "health_timeout": float(os.environ.get("HEALTH_TIMEOUT", "3000")),
            "tag": "qwen3-5-mmmupro-e2e",
        },
        "skip_unless": (
            os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1",
            "multi-host live-server not provided (the JobSet launches the server)",
        ),
        "gates": [
            {
                "dataset": "mmmu_pro",
                "label": "mmmu_pro",
                "report_only": True,
                "runner": make_runner(QWEN3_5_397B, "mmmu_pro"),
            },
        ],
    }


if __name__ == "__main__":
    model_config_main(build_model_config(), base_url=BASE_URL)
