"""Qwen3-Coder-480B EAGLE3 single-host GSM8K quality gates.

The quality lanes cover the standard and hybrid-DP speculative-decoding
configurations, then score 200 GSM8K examples through run_eval.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

register_tpu_ci(est_time=7200, suite="nightly-tpu-4-v7", nightly=True)

SERVED_MODEL_NAME = "qwen3-coder-480b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "3600"))

GSM8K_ACCURACY = 0.65

LAUNCH_ENV = {
    "MODEL": "/weights/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "DRAFT_MODEL": (
        "/weights/" "SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-SpecForge-EigenAI"
    ),
    "ISL": "4096",
    "OSL": "512",
    "CONCUR": "16",
    "CONTEXT_LEN": "8192",
    "MAX_TOTAL_TOKENS": "8192",
    "ATTENTION_BACKEND": "pallas_tpu_rpa_v3",
    "TPU_COMPILE_TOKENS": "512 2048 4096",
    "TPU_COMPILE_BS": "1 2 4 16",
    "DISABLE_RADIX_CACHE": "0",
    "ENABLE_PROFILER_ENDPOINT": "0",
    "EXTRA_ARGS": (
        f"--served-model-name {SERVED_MODEL_NAME} "
        "--speculative-algorithm EAGLE3 "
        "--speculative-draft-model-path "
        "/weights/"
        "SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-SpecForge-EigenAI "
        "--speculative-num-steps 3 "
        "--speculative-eagle-topk 1 "
        "--speculative-num-draft-tokens 4 "
        "--tpu-compile-tracer dynamo"
    ),
}

HYBRID_DP_LAUNCH_ENV = {
    **LAUNCH_ENV,
    "DP_SIZE": "2",
    # These are DP-local buckets; launcher reconciliation adds global bs=16.
    "TPU_COMPILE_BS": "1 2 4 8",
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "qwen3-coder-480b-spec-decoding-gsm8k",
    "variant": "TP8+EP8+EAGLE3",
    "launch": {
        "recipe": "recipe/qwen3_coder_480b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "qwen3-coder-480b-spec-gsm8k-e2e",
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

HYBRID_DP_MODEL_CONFIG = {
    **MODEL_CONFIG,
    "test_name": "qwen3-coder-480b-spec-decoding-hybrid-dp-gsm8k",
    "variant": "TP8+EP8+DP2+EAGLE3",
    "launch": {
        **MODEL_CONFIG["launch"],
        "env": HYBRID_DP_LAUNCH_ENV,
        "tag": "qwen3-coder-480b-spec-dp-gsm8k-e2e",
    },
}


if __name__ == "__main__":
    model_config_main([MODEL_CONFIG, HYBRID_DP_MODEL_CONFIG], base_url=BASE_URL)
