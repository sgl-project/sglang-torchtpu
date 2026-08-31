"""GPT-OSS 120B TPU AIME 2025 gate."""

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import aime25_openai_runner, model_config_main
from sglang.test.tpu.gpt_oss_120b_eval_common import (
    BASE_URL,
    SERVED_MODEL_NAME,
    make_launch_env,
    make_model_config,
)

register_tpu_ci(est_time=4500, suite="tpu-eval-gpt-oss-120b-aime", nightly=True)

MAX_TOKENS = 65536

LAUNCH_ENV = make_launch_env(
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
        model=SERVED_MODEL_NAME,
        n_repeats=3,
        reasoning_effort="high",
        max_tokens=MAX_TOKENS,
        n_threads=16,
    ),
    "baseline": 0.90,
    "atol": 0.00,
}

MODEL_CONFIG = make_model_config(
    test_name="gpt-oss-120b-aime",
    tag="aime-e2e",
    launch_env=LAUNCH_ENV,
    gate=GATE,
)

if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
