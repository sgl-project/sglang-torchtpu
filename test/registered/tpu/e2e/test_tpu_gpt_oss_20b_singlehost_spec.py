"""GPT-OSS-20B single-host EAGLE3 PR gate."""

import os

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

register_tpu_ci(
    est_time=650,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "gpt-oss-20b", "spec"],
)

SERVED_MODEL_NAME = "gpt-oss-20b"
MODEL_PATH = resolve_tpu_model_path("openai/gpt-oss-20b")
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))
DRAFT_MODEL_PATH = resolve_tpu_model_path("zhuyksir/EAGLE3-gpt-oss-20b-bf16")
PROMPT = "What is the capital of France?"


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
    num_accept_tokens = correct_drafts + verify_count
    accept_rate = correct_drafts / proposed if proposed > 0 else 0.0
    print(
        "Speculative decoding metrics: "
        f"spec_verify_ct={verify_count}, "
        f"num_accept_tokens={num_accept_tokens}, "
        f"num_proposed_drafts={proposed}, "
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
    "DRAFT_MODEL": DRAFT_MODEL_PATH,
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
        "--speculative-algorithm EAGLE3 "
        f"--speculative-draft-model-path {DRAFT_MODEL_PATH} "
        "--speculative-num-steps 3 "
        "--speculative-eagle-topk 1 "
        "--speculative-num-draft-tokens 4 "
        "--trust-remote-code"
    ),
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-20b-spec-decoding",
    "variant": "TP8+EP8+EAGLE3",
    "launch": {
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-20b-spec-e2e",
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
