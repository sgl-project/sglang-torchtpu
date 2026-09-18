"""GPT-OSS-20B single-host EAGLE3 PR gate."""

import os
import sys
from copy import deepcopy
from pathlib import Path

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.server_test_suite import resolve_tpu_model_path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmark.tpu.recipe.configs.gpt_oss_120b.server import CONFIG

register_tpu_ci(
    est_time=1800,
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


LAUNCH_CONFIG = deepcopy(CONFIG)
LAUNCH_CONFIG["server_args"].update(
    {
        "context_length": 4096,
        "max_running_requests": 4,
        "torch_compile_max_bs": 4,
        "chunked_prefill_size": min(max(2048, 2048), 8192),
        "max_prefill_tokens": min(max(2048, 2048), 8192),
        "max_total_tokens": 8192,
        "model_path": MODEL_PATH,
        "tpu_compile_tokens": [512, 1024, 2048],
        "tpu_compile_bs": [1, 2, 4],
        "reasoning_parser": "gpt-oss",
        "served_model_name": SERVED_MODEL_NAME,
        "speculative_algorithm": "EAGLE3",
        "speculative_draft_model_path": DRAFT_MODEL_PATH,
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
        "trust_remote_code": True,
    }
)

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-20b-spec-decoding",
    "variant": "TP8+EP8+EAGLE3",
    "launch": {
        **LAUNCH_CONFIG,
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
