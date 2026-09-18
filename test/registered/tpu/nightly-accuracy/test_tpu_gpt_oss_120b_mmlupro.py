"""GPT-OSS 120B TPU MMLU-Pro nightly score."""

from copy import deepcopy

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.gpt_oss_120b_eval_common import (
    BASE_URL,
    CONFIG,
    make_model_config,
)

register_tpu_ci(
    est_time=4500,
    suite="nightly-tpu-accuracy-gpt-oss-120b-mmlupro",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "accuracy"],
)

LAUNCH_CONFIG = deepcopy(CONFIG)
LAUNCH_CONFIG["server_args"].update(
    {
        "max_running_requests": 32,
        "torch_compile_max_bs": 32,
        "context_length": 36864,
        "max_total_tokens": 1179648,
        "tpu_compile_bs": [1, 2, 4, 8, 16, 32],
    }
)

GATE = {
    "dataset": "mmlu_pro",
    "label": "mmlu_pro_high",
    "lm_eval_config": "test/lm_eval_configs/GPT-OSS-120B-TPU-MMLU-Pro.yaml",
    "report_only": True,
}

MODEL_CONFIG = make_model_config(
    test_name="gpt-oss-120b-mmlupro",
    series_id="accuracy-gpt-oss-120b-mmlupro",
    tag="mmlupro-e2e",
    launch_config=LAUNCH_CONFIG,
    gate=GATE,
)

if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
