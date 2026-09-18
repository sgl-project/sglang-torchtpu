"""GPT-OSS-120B EAGLE3 single-host GSM8K quality gate.

The quality lane launches the same speculative-decoding configuration as the
functional integration test, then scores 200 GSM8K examples through run_eval.
"""

import os
import sys
from copy import deepcopy
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmark.tpu.recipe.configs.gpt_oss_120b.server import CONFIG

register_tpu_ci(
    est_time=3600,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "spec"],
)

SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

GSM8K_ACCURACY = 0.70  # measured 0.7850 over 200 examples
SPEC_ACCEPT_LENGTH_THRESHOLD = float(
    os.environ.get("SPEC_ACCEPT_LENGTH_THRESHOLD", "1.90")
)


LAUNCH_CONFIG = deepcopy(CONFIG)
LAUNCH_CONFIG["server_args"].update(
    {
        "context_length": 8192,
        "max_running_requests": 16,
        "torch_compile_max_bs": 16,
        "chunked_prefill_size": min(max(4096, 2048), 8192),
        "max_prefill_tokens": min(max(4096, 2048), 8192),
        "max_total_tokens": 8192,
        "model_path": "/weights/gpt-oss-120b",
        "tpu_compile_tokens": [512, 2048, 4096],
        "tpu_compile_bs": [1, 2, 4, 16],
        "reasoning_parser": "gpt-oss",
        "served_model_name": SERVED_MODEL_NAME,
        "speculative_algorithm": "EAGLE3",
        "speculative_draft_model_path": "lmsys/EAGLE3-gpt-oss-120b-bf16",
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
    }
)

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-spec-decoding-gsm8k",
    "variant": "TP8+EP8+EAGLE3",
    "launch": {
        **LAUNCH_CONFIG,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-spec-gsm8k-e2e",
    },
    "gates": [
        {
            "dataset": "gsm8k",
            "label": "gsm8k",
            "baseline": GSM8K_ACCURACY,
            "spec_accept_length_threshold": SPEC_ACCEPT_LENGTH_THRESHOLD,
            "num_examples": 200,
            "num_threads": 16,
            "max_tokens": 512,
            "temperature": 0.0,
        }
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
