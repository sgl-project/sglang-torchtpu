"""GPT-OSS-120B EAGLE3 DP-attention GSM8K quality gate."""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

register_tpu_ci(est_time=3600, suite="nightly-tpu-4-v7", nightly=True)

SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

GSM8K_ACCURACY = 0.70  # measured 0.7850 over 200 examples

LAUNCH_ENV = {
    "MODEL": "/weights/gpt-oss-120b",
    "DRAFT_MODEL": "lmsys/EAGLE3-gpt-oss-120b-bf16",
    "ISL": "4096",
    "OSL": "512",
    "CONCUR": "16",
    "CONTEXT_LEN": "8192",
    "TPU_COMPILE_TOKENS": "512 1024 2048 4096",
    # DP-local buckets: DP4 turns the global concurrency of 16 into four per rank.
    "TPU_COMPILE_BS": "1 2 4",
    "MAX_TOTAL_TOKENS": "8192",
    "ENABLE_PROFILER_ENDPOINT": "0",
    "DP_SIZE": "4",
    # Keep speculative recompilation enabled until distributed top-k precompiles
    # the same token buckets as the decoder.
    "SGLANG_TPU_ALLOW_SPEC_RECOMPILE": "1",
    "EXTRA_ARGS": (
        "--reasoning-parser gpt-oss "
        f"--served-model-name {SERVED_MODEL_NAME} "
        "--speculative-algorithm EAGLE3 "
        "--speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16 "
        "--speculative-num-steps 3 "
        "--speculative-eagle-topk 1 "
        "--speculative-num-draft-tokens 4 "
        "--tpu-compile-tracer dynamo"
    ),
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-spec-dpa-gsm8k",
    "variant": "TP8+EP8+DP4+EAGLE3",
    "launch": {
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-spec-dpa-gsm8k-e2e",
    },
    "gates": [
        {
            "dataset": "gsm8k",
            "label": "gsm8k",
            "baseline": GSM8K_ACCURACY,
            "num_examples": 200,
            "num_threads": 16,
            "max_tokens": 512,
            "temperature": 0.0,
        }
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
