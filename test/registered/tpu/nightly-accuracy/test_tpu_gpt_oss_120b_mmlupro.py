"""GPT-OSS 120B TPU MMLU-Pro nightly score."""

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.gpt_oss_120b_eval_common import (
    BASE_URL,
    make_launch_env,
    make_model_config,
)

register_tpu_ci(est_time=4500, suite="tpu-eval-gpt-oss-120b-mmlupro", nightly=True)

LAUNCH_ENV = make_launch_env(
    MAX_TOTAL_TOKENS="1179648",
    OSL="32768",
    CONCUR="32",
    CONTEXT_LEN="36864",
    TPU_COMPILE_BS="1 2 4 8 16 32",
)

GATE = {
    "dataset": "mmlu_pro",
    "label": "mmlu_pro_high",
    "lm_eval_config": "test/lm_eval_configs/GPT-OSS-120B-TPU-MMLU-Pro.yaml",
    "report_only": True,
}

MODEL_CONFIG = make_model_config(
    test_name="gpt-oss-120b-mmlupro",
    tag="mmlupro-e2e",
    launch_env=LAUNCH_ENV,
    gate=GATE,
)

if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
