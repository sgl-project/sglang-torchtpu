"""GPT-OSS-120B single-host DFlash PR gate.

The nightly lane scores GSM8K; this one only proves DFlash proposes and accepts
draft tokens, so the PR gate catches a broken draft path without paying for an
eval.
"""

import os

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

register_tpu_ci(
    est_time=1800,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "gpt-oss-120b", "spec", "dflash"],
)

SERVED_MODEL_NAME = "gpt-oss-120b"
MODEL_PATH = resolve_tpu_model_path("openai/gpt-oss-120b")
DRAFT_MODEL_PATH = resolve_tpu_model_path("z-lab/gpt-oss-120b-DFlash")
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))
PROMPT = "What is the capital of France?"

# DFlash drafts one fixed-size block per verify; the arg hook pins num_steps and
# eagle_topk to 1, so the block size is the only speculative knob that applies.
DFLASH_BLOCK_SIZE = 6


def spec_decoding_runner(base_url: str):
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json={
            "model": SERVED_MODEL_NAME,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 64,
            "temperature": 0.0,
            "return_meta_info": True,
        },
        timeout=120,
    )
    response.raise_for_status()
    meta = response.json()["choices"][0]["meta_info"]
    verify_count = meta.get("spec_verify_ct", 0)
    proposed = meta.get("spec_num_proposed_drafts", 0)
    correct_drafts = meta.get("spec_num_correct_drafts", 0)
    accept_rate = correct_drafts / proposed if proposed > 0 else 0.0
    print(
        "Speculative decoding metrics: "
        f"spec_verify_ct={verify_count}, "
        f"num_proposed_drafts={proposed}, "
        f"num_correct_drafts={correct_drafts}, "
        f"accept_rate={accept_rate:.4f}"
    )
    failures = []
    if verify_count <= 0:
        failures.append(f"speculative verification did not run: {verify_count=}")
    if proposed <= 0:
        failures.append(f"no draft tokens were proposed: {proposed=}")
    if correct_drafts <= 0:
        failures.append(f"no draft tokens were correct: {correct_drafts=}")
    return {
        "score": correct_drafts,
        "_passed": not failures,
        "_failures": failures,
    }


LAUNCH_ENV = {
    "MODEL": MODEL_PATH,
    "ISL": "2048",
    "OSL": "512",
    "CONCUR": "4",
    "CONTEXT_LEN": "4096",
    "MAX_TOTAL_TOKENS": "8192",
    "TPU_COMPILE_TOKENS": "512 1024 2048",
    "TPU_COMPILE_BS": "1 2 4",
    "ENABLE_PROFILER_ENDPOINT": "0",
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
    "test_name": "gpt-oss-120b-dflash-smoke",
    "variant": "TP8+EP8+DFLASH",
    "launch": {
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-120b-dflash-smoke-e2e",
    },
    "gates": [
        {
            "dataset": "speculative_decoding",
            "label": "speculative-decoding",
            "runner": spec_decoding_runner,
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
