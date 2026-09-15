"""GPT-OSS 120B TPU full-dataset GSM8K nightly score."""

from copy import deepcopy

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.gpt_oss_120b_eval_common import (
    BASE_URL,
    CONFIG,
    make_model_config,
)

register_tpu_ci(
    est_time=2700,
    suite="nightly-tpu-accuracy-gpt-oss-120b-gsm8k",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "accuracy"],
)

LAUNCH_CONFIG = deepcopy(CONFIG)
LAUNCH_CONFIG["server_args"].update(
    {
        "max_running_requests": 32,
        "torch_compile_max_bs": 32,
        "context_length": 8192,
        "max_total_tokens": 262144,
        "tpu_compile_bs": [8, 16, 32],
    }
)

GATE = {
    "dataset": "gsm8k",
    "label": "gsm8k_full",
    "report_only": True,
    "num_examples": 1319,
    "num_threads": 32,
    "max_tokens": 4096,
    "temperature": 0.0,
    "reasoning_effort": "low",
}

MODEL_CONFIG = make_model_config(
    test_name="gpt-oss-120b-gsm8k",
    series_id="accuracy-gpt-oss-120b-gsm8k",
    tag="gsm8k-e2e",
    launch_config=LAUNCH_CONFIG,
    gate=GATE,
)

if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
