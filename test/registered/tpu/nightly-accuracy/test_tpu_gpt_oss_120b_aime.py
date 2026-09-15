"""GPT-OSS 120B TPU AIME 2025 nightly score."""

from copy import deepcopy

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import aime25_openai_runner, model_config_main
from sglang.test.tpu.gpt_oss_120b_eval_common import (
    BASE_URL,
    CONFIG,
    SERVED_MODEL_NAME,
    make_model_config,
)

register_tpu_ci(
    est_time=4500,
    suite="nightly-tpu-accuracy-gpt-oss-120b-aime",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "accuracy"],
)

MAX_TOKENS = 65536

LAUNCH_CONFIG = deepcopy(CONFIG)
LAUNCH_CONFIG["server_args"].update(
    {
        "max_running_requests": 16,
        "torch_compile_max_bs": 16,
        "context_length": 66560,
        "max_total_tokens": 1064960,
        "tpu_compile_bs": [1, 2, 4, 8, 16],
    }
)

GATE = {
    "dataset": "aime25",
    "label": "aime25_high",
    "runner": aime25_openai_runner(
        model=SERVED_MODEL_NAME,
        n_repeats=3,
        reasoning_effort="high",
        max_tokens=MAX_TOKENS,
        n_threads=16,
    ),
    "report_only": True,
}

MODEL_CONFIG = make_model_config(
    test_name="gpt-oss-120b-aime",
    series_id="accuracy-gpt-oss-120b-aime",
    tag="aime-e2e",
    launch_config=LAUNCH_CONFIG,
    gate=GATE,
)

if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
