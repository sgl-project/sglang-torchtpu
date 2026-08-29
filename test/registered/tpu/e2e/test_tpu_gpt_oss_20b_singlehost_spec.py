"""GPT-OSS-20B single-host EAGLE3 PR gate."""

import os

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.server_test_suite import resolve_tpu_model_path

register_tpu_ci(est_time=1800, suite="stage-b-test-tpu-4-v7")

SERVED_MODEL_NAME = "gpt-oss-20b"
MODEL_PATH = resolve_tpu_model_path("openai/gpt-oss-20b")
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))
DRAFT_MODEL_PATH = resolve_tpu_model_path("zhuyksir/EAGLE3-gpt-oss-20b-bf16")


def spec_decoding_runner(base_url: str):
    response = requests.post(
        f"{base_url.rstrip('/')}/generate",
        json={
            "text": "The capital of France is",
            "sampling_params": {"max_new_tokens": 64, "temperature": 0.0},
        },
        timeout=120,
    )
    response.raise_for_status()
    meta = response.json()["meta_info"]
    verify_count = meta.get("spec_verify_ct", 0)
    proposed = meta.get("spec_num_proposed_drafts", 0)
    accepted = meta.get("spec_num_correct_drafts", 0)
    failures = []
    if verify_count <= 0:
        failures.append(f"speculative verification did not run: {verify_count=}")
    if proposed <= 0:
        failures.append(f"no draft tokens were proposed: {proposed=}")
    if accepted <= 0:
        failures.append(f"no draft tokens were accepted: {accepted=}")
    return {
        "score": accepted,
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
    "SGLANG_TPU_ALLOW_SPEC_RECOMPILE": "1",
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
