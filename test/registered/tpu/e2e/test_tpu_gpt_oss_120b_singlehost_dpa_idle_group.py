"""GPT-OSS-120B single-host DPA DP2/attention-TP4 gate at one request per group.

Attention DP pads every decode to the attention-TP width, four rows above the
request cap, and a group idles beside the other whenever its request finishes
first.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.utils.accuracy_base import recipe_server_args

register_tpu_ci(
    est_time=3600,
    suite="external-nightly-1host",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "dpa", "hybrid"],
)

SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

GSM8K_ACCURACY = 0.70
CONFIG_MODULE = "recipe/configs/gpt_oss_120b/dpa_idle_group.py"
# One request per DP group, so the client has to follow the recipe.
DP_SIZE = recipe_server_args(CONFIG_MODULE)["dp_size"]


MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-dpa-idle-group-gsm8k",
    "variant": "TP8+EP8+DP2(attn-tp4)",
    "launch": {
        "config": CONFIG_MODULE,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-dpa-idle-group-gsm8k-e2e",
    },
    "gates": [
        {
            "dataset": "gsm8k",
            "label": "gsm8k",
            "baseline": GSM8K_ACCURACY,
            "num_examples": 200,
            "num_threads": DP_SIZE,
            "max_tokens": 512,
            "temperature": 0.0,
        }
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
