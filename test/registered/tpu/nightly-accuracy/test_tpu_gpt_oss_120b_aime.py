"""GPT-OSS 120B TPU AIME 2025 nightly score."""

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import (
    BASE_URL,
    GPT_OSS_120B,
    aime25_openai_runner,
    make_launch_env,
    make_model_config,
    model_config_main,
)

register_tpu_ci(
    est_time=4500,
    suite="nightly-tpu-accuracy-gpt-oss-120b-aime",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "accuracy"],
)

MAX_TOKENS = 65536

LAUNCH_ENV = make_launch_env(
    GPT_OSS_120B,
    MAX_TOTAL_TOKENS="1064960",
    OSL=str(MAX_TOKENS),
    CONCUR="16",
    CONTEXT_LEN="66560",
    TPU_COMPILE_BS="1 2 4 8 16",
)

GATE = {
    "dataset": "aime25",
    "label": "aime25_high",
    "runner": aime25_openai_runner(
        model=GPT_OSS_120B["served_model_name"],
        n_repeats=3,
        reasoning_effort="high",
        max_tokens=MAX_TOKENS,
        n_threads=16,
    ),
    "report_only": True,
}

MODEL_CONFIG = make_model_config(
    GPT_OSS_120B,
    test_name="gpt-oss-120b-aime",
    series_id="accuracy-gpt-oss-120b-aime",
    tag="aime-e2e",
    launch_env=LAUNCH_ENV,
    gate=GATE,
)

if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
