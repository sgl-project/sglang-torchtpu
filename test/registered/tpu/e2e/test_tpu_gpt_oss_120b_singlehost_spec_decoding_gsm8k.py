"""GPT-OSS-120B EAGLE3 single-host GSM8K quality gates.

The quality lanes cover the standard and hybrid-DP speculative-decoding
configurations, then score 200 GSM8K examples through run_eval.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

register_tpu_ci(est_time=7200, suite="nightly-tpu-4-v7", nightly=True)

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
    "TPU_COMPILE_TOKENS": "512 2048 4096",
    "TPU_COMPILE_BS": "1 2 4 16",
    "MAX_TOTAL_TOKENS": "8192",
    "ENABLE_PROFILER_ENDPOINT": "0",
    # TODO(tpu): remove once the TPU distributed top-k path is precompiled for
    # the same token buckets as the decoder. Today it is not, and the mismatch
    # kills the run outright: TPU_COMPILE_TOKENS above asks for 512/2048/4096 and
    # the decoder policy honours it, but logits_processor's
    # get_tpu_distributed_top_k_for_batch guards on its own shapes and has only
    # seen 512 and 4096. The first real request at 2048 tokens trips
    #
    #   RuntimeError: Detected recompile when torch.compile stance is
    #   'fail_on_recompile' ... size mismatch at index 0. expected 4096, actual 2048
    #
    # which raises inside run_batch, takes every TP rank's scheduler down, and
    # leaves the eval retrying against a dead server -- gsm8k scores 0.0 with no
    # hint that the cause was a compile guard. Observed on run 32692936982.
    #
    # This flag is the escape hatch scheduler._maybe_arm_fail_on_recompile already
    # provides for speculative decoding: recompile instead of dying. It costs tens
    # of seconds on the first unseen shape (est_time has room) and it does NOT
    # hide the bug -- the recompile is still logged, and the real fix is to make
    # the top-k path cover the decoder's buckets.
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

HYBRID_DP_LAUNCH_ENV = {
    **LAUNCH_ENV,
    "DP_SIZE": "2",
    # These are DP-local buckets; launcher reconciliation adds global bs=16.
    "TPU_COMPILE_BS": "1 2 4 8",
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b-spec-decoding-gsm8k",
    "variant": "TP8+EP8+EAGLE3",
    "launch": {
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-spec-gsm8k-e2e",
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
    "test_name": "gpt-oss-120b-spec-decoding-hybrid-dp-gsm8k",
    "variant": "TP8+EP8+DP2+EAGLE3",
    "launch": {
        **MODEL_CONFIG["launch"],
        "env": HYBRID_DP_LAUNCH_ENV,
        "tag": "gpt-oss-spec-dp-gsm8k-e2e",
    },
}


if __name__ == "__main__":
    model_config_main([MODEL_CONFIG, HYBRID_DP_MODEL_CONFIG], base_url=BASE_URL)
