"""GPT-OSS 120B TPU GPQA Diamond nightly score."""

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import gpqa_openai_runner, model_config_main
from sglang.test.tpu.gpt_oss_120b_eval_common import (
    BASE_URL,
    SERVED_MODEL_NAME,
    make_launch_env,
    make_model_config,
)

register_tpu_ci(est_time=4500, suite="tpu-eval-gpt-oss-120b-gpqa", nightly=True)

MAX_TOKENS = 65536

LAUNCH_ENV = make_launch_env(
    MAX_TOTAL_TOKENS="1114112",
    OSL=str(MAX_TOKENS),
    CONCUR="16",
    CONTEXT_LEN="69632",
    TPU_COMPILE_BS="1 2 4 8 16",
)

GATE = {
    "dataset": "gpqa",
    "label": "gpqa_diamond_high",
    "runner": gpqa_openai_runner(
        model=SERVED_MODEL_NAME,
        n_repeats=1,
        reasoning_effort="high",
        max_tokens=MAX_TOKENS,
        n_threads=16,
        num_examples=198,
    ),
    "report_only": True,
}

MODEL_CONFIG = make_model_config(
    test_name="gpt-oss-120b-gpqa",
    tag="gpqa-e2e",
    launch_env=LAUNCH_ENV,
    gate=GATE,
)

if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
