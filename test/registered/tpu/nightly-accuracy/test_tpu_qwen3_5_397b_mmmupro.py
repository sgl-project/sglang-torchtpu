"""Qwen3.5-397B-A17B TPU MMMU-Pro nightly score, all 1,730 standard-10 questions.

The JobSet brings recipe/qwen3_5_397b_mh.sh up on both hosts; rank 0 runs this
file against it. Thinking on, with the model card's sampling.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST
from sglang.test.tpu import model_config_main, sgl_eval_runner

register_tpu_ci(
    est_time=9000,
    suite="nightly-tpu-multihost-accuracy",
    nightly=True,
    labels=["multihost", "qwen3.5-397b", "accuracy"],
)

MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/configs/qwen3_5_397b/server.py",
    "weights_pvc": "model-weights",
    # Measured 2 h 13 min for the full set at 128 concurrent requests.
    # Which accuracy-test-tpu.yml lane runs this file.
    "schedule": "mmmu-gemma4-qwen35",
    "timeout_per_file": 10800,
}

MODEL_PATH = "/weights/Qwen3.5-397B-A17B"
BASE_URL = os.environ.get("BASE_URL", DEFAULT_URL_FOR_TEST)
MODEL_CONFIG = {
    "id": "accuracy-qwen3-5-397b-mmmupro",
    "display_name": "Qwen3.5 397B-A17B",
    "dashboard": "accuracy",
    "dashboard_model": "qwen3.5-397b",
    "model": MODEL_PATH,
    "test_name": "Qwen3.5-397B-multihost-mmmupro",
    "variant": "2host-TP16+EP16-bf16",
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
            "runner": sgl_eval_runner(
                "mmmu_pro",
                model=MODEL_PATH,
                max_tokens=32768,
                num_threads=128,
                temperature=0.6,
                top_p=0.95,
                # The Qwen3.5 template reads enable_thinking; --thinking sets `thinking`.
                chat_template_kwargs={"enable_thinking": True},
            ),
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
