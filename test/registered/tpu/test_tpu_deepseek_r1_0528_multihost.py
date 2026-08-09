"""DeepSeek-R1-0528 (671B MoE, fp8) multi-host e2e accuracy: GSM8K.

The JobSet launches the server (recipe/deepseek_r1_0528_mh.sh, 2 hosts, tp=16); this
file attaches to it and scores one gate:

  GSM8K  0.960  run_eval, 100 questions, single pass
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST
from sglang.test.tpu import model_config_main

register_tpu_ci(est_time=7200, suite="nightly-tpu-multihost", nightly=True)

MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/deepseek_r1_0528_mh.sh",
    "weights_pvc": "model-weights-rox",
}

MODEL_PATH = os.environ.get("MODEL", "/weights/DeepSeek-R1-0528")
BASE_URL = os.environ.get("BASE_URL", DEFAULT_URL_FOR_TEST)
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "3000"))

CARD_TEMPERATURE = 0.6
CARD_TOP_P = 0.95

GSM8K_GROUND_TRUTH = 0.96
ATOL = 0.03

NUM_EXAMPLES = 100
NUM_THREADS = 128
MAX_TOKENS = 8192


MODEL_CONFIG = {
    "model": MODEL_PATH,
    "test_name": "DeepSeek-R1-0528-multihost",
    "variant": "2host-TP16+EP16",
    "launch": {"external": True, "health_timeout": HEALTH_TIMEOUT, "tag": "r1-e2e"},
    "skip_unless": (
        os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1",
        "multi-host live-server not provided (the JobSet launches the server)",
    ),
    "gates": [
        {
            "dataset": "gsm8k",
            "label": "gsm8k",
            "baseline": GSM8K_GROUND_TRUTH,
            "atol": ATOL,
            "num_examples": NUM_EXAMPLES,
            "num_threads": NUM_THREADS,
            "temperature": CARD_TEMPERATURE,
            "top_p": CARD_TOP_P,
            "max_tokens": MAX_TOKENS,
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
