"""GPT-OSS-120B DFlash single-host GSM8K quality gate.

The quality lane launches the same speculative-decoding configuration as the
functional integration test, then scores 200 GSM8K examples through run_eval.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

register_tpu_ci(
    est_time=3600,
    suite="external-nightly-1host",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "spec", "dflash"],
)

SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

MODEL_PATH = os.environ.get("MODEL", "/weights/gpt-oss-120b")
DRAFT_MODEL_PATH = os.environ.get("DRAFT_MODEL", "z-lab/gpt-oss-120b-DFlash")

# Fixed by the z-lab DFlash checkpoint; the arg hook pins num_steps and
# eagle_topk to 1, so block size is the only speculative knob that applies.
DFLASH_BLOCK_SIZE = 6

GSM8K_ACCURACY = 0.70  # measured 0.8300 over 200 examples
# Measured 3.244 aggregate over the same 200 examples (per-request 2.55..4.08).
SPEC_ACCEPT_LENGTH_THRESHOLD = float(
    os.environ.get("SPEC_ACCEPT_LENGTH_THRESHOLD", "3.0")
)

LAUNCH_ENV = {
    "MODEL": MODEL_PATH,
    "DRAFT_MODEL": DRAFT_MODEL_PATH,
    "ISL": "4096",
    "OSL": "512",
    "CONCUR": "16",
    "CONTEXT_LEN": "8192",
    "TPU_COMPILE_TOKENS": "512 2048 4096",
    "TPU_COMPILE_BS": "1 2 4 16",
    "MAX_TOTAL_TOKENS": "8192",
    "MEM_FRACTION_STATIC": os.environ.get("MEM_FRACTION_STATIC", "0.6"),
    "ENABLE_PROFILER_ENDPOINT": "0",
    # Broadcast only the sites that can genuinely differ across ranks -- memory
    # init and the RNG draws. The deterministic ones (greedy accept) compute the
    # same answer everywhere, so syncing them costs a collective per step.
    "SGLANG_SPEC_TP_SYNC": "rng",
    "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER": "1",
    "EXTRA_ARGS": (
        "--reasoning-parser gpt-oss "
        f"--served-model-name {SERVED_MODEL_NAME} "
        "--speculative-algorithm DFLASH "
        f"--speculative-draft-model-path {DRAFT_MODEL_PATH} "
        f"--speculative-dflash-block-size {DFLASH_BLOCK_SIZE}"
    ),
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-dflash-gsm8k",
    "variant": "TP8+EP8+DFLASH",
    "launch": {
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-dflash-gsm8k-e2e",
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
