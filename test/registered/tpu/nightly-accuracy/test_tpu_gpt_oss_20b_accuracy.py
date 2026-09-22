"""GPT-OSS-20B: four concurrent AIME25 repeats; GPQA and MMLU-250 once."""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.server_test_suite import resolve_tpu_model_path

register_tpu_ci(
    est_time=7200,
    suite="nightly-tpu-accuracy-gpt-oss-20b",
    nightly=True,
    labels=["singlehost", "gpt-oss-20b", "accuracy"],
)

MODEL_CONFIG = {
    "model": "gpt-oss-20b",
    "test_name": "gpt-oss-20b-accuracy",
    "id": "accuracy-gpt-oss-20b",
    "display_name": "GPT-OSS 20B",
    "variant": "TP8+EP8",
    "dashboard": "accuracy",
    "dashboard_model": "gpt-oss-20b",
    "launch": {
        "recipe": "recipe/gpt_oss_20b_accuracy.sh",
        "env": {"MODEL": resolve_tpu_model_path("openai/gpt-oss-20b")},
        "health_timeout": 3000,
        "tag": "gpt-oss-20b-accuracy",
    },
    "gates": [
        {
            "dataset": "gpqa",
            "label": "gpqa_diamond_high",
            "report_only": True,
            "repeat": 1,
            "num_threads": 128,
            "reasoning_effort": "high",
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 131072,
        },
        {
            "dataset": "aime25",
            "label": "aime25_high",
            "report_only": True,
            # sgl-eval submits all 30 x 4 samples to one shared worker pool.
            "repeat": 4,
            "num_threads": 128,
            "reasoning_effort": "high",
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 131072,
        },
        {
            "dataset": "mmlu",
            "label": "mmlu_high",
            "report_only": True,
            "repeat": 1,
            "num_examples": 250,
            "num_threads": 128,
            "reasoning_effort": "high",
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 131072,
        },
    ],
}

if __name__ == "__main__":
    model_config_main(
        MODEL_CONFIG, base_url=os.environ.get("BASE_URL", "http://127.0.0.1:30000")
    )
