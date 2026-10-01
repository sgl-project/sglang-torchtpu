"""GPT-OSS 120B TPU full-dataset GSM8K nightly score."""

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import (
    BASE_URL,
    GPT_OSS_120B,
    make_launch_env,
    make_model_config,
    model_config_main,
)

register_tpu_ci(
    est_time=2700,
    suite="nightly-tpu-accuracy-gpt-oss-120b-gsm8k",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "accuracy"],
)

LAUNCH_ENV = make_launch_env(
    GPT_OSS_120B,
    MAX_TOTAL_TOKENS="262144",
    OSL="4096",
    CONCUR="32",
    CONTEXT_LEN="8192",
    TPU_COMPILE_BS="8 16 32",
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
    GPT_OSS_120B,
    test_name="gpt-oss-120b-gsm8k",
    series_id="accuracy-gpt-oss-120b-gsm8k",
    tag="gsm8k-e2e",
    launch_env=LAUNCH_ENV,
    gate=GATE,
)

if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
