"""Qwen3-Coder-30B-A3B-Instruct-FP8 single-host e2e accuracy via lm-eval.

Qwen3-Coder-30B-A3B-Instruct-FP8-TPU.yaml is the single source of truth for the
mmlu_llama / mmlu_pro ground truths and tolerances. The gate runs on lm-eval because
mmlu_pro has no simple-evals equivalent, and mmlu_llama's few-shot protocol is what
the reference values were measured with.

Serving config lives in recipe/qwen3_coder_30b_a3b_fp8.sh, which CI runs verbatim.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

register_tpu_ci(
    est_time=3600,
    suite="stage-b-test-tpu-4-v7",
    disabled="TPU PR runners are not provisioned yet",
)

MODEL_PATH = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
LM_EVAL_CONFIG = "test/lm_eval_configs/Qwen3-Coder-30B-A3B-Instruct-FP8-TPU.yaml"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))


MODEL_CONFIG = {
    "model": MODEL_PATH,
    "test_name": "Qwen3-Coder-30B-A3B-Instruct-FP8",
    "variant": "TP8+EP8",
    "launch": {
        "recipe": "recipe/qwen3_coder_30b_a3b_fp8.sh",
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "qwen3-coder-e2e",
    },
    "gates": [
        {
            "dataset": "mmlu_llama+mmlu_pro",
            "label": "lm_eval",
            "lm_eval_config": LM_EVAL_CONFIG,
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
