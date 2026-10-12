"""GPT-OSS-20B EAGLE3 + DP-attention PR gate on one TPU host.

This is the only per-commit test that runs a server with DP attention
(`DP_SIZE` > 1 becomes `--dp-size N --enable-dp-attention`), so it stays on the
gate. It deliberately does NOT set `--tpu-compile-tracer dynamo`: the tracers
have their own gate in test_tpu_qwen3_0_6b_singlehost_compile_tracer.py at 150s
against Qwen3-0.6B, and carrying the flag here made this launch the second most
expensive in the e2e suite.
"""

import os

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

register_tpu_ci(est_time=1800, suite="stage-b-test-tpu-4-v7")

SERVED_MODEL_NAME = "gpt-oss-20b"
MODEL_PATH = resolve_tpu_model_path("openai/gpt-oss-20b")
DRAFT_MODEL_PATH = resolve_tpu_model_path("zhuyksir/EAGLE3-gpt-oss-20b-bf16")
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

DP_SIZE = 2
PROMPT = "What is the capital of France?"


def spec_dpa_runner(base_url: str):
    """Validate EAGLE3 execution and acceptance on every routed DP rank."""
    base_url = base_url.rstrip("/")
    server_info_response = requests.get(f"{base_url}/server_info", timeout=30)
    server_info_response.raise_for_status()
    server_info = server_info_response.json()

    failures = []
    for key in ("moe_runner_backend", "speculative_moe_runner_backend"):
        if server_info.get(key) != "pallas_tpu":
            failures.append(f"{key}={server_info.get(key)!r}, expected 'pallas_tpu'")

    total_correct_drafts = 0
    for dp_rank in range(DP_SIZE):
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": SERVED_MODEL_NAME,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": 64,
                "temperature": 0.0,
                "return_meta_info": True,
                "routed_dp_rank": dp_rank,
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
            f"Speculative decoding metrics for dp_rank={dp_rank}: "
            f"spec_verify_ct={verify_count}, "
            f"num_accept_tokens={num_accept_tokens}, "
            f"num_proposed_drafts={proposed}, "
            f"accept_rate={accept_rate:.4f}"
        )

        if meta.get("dp_rank") != dp_rank:
            failures.append(f"requested dp_rank={dp_rank}, got {meta.get('dp_rank')!r}")
        if verify_count <= 0:
            failures.append(f"dp_rank={dp_rank}: no spec verifications")
            continue
        if proposed <= 0:
            failures.append(f"dp_rank={dp_rank}: no proposed drafts")
        if correct_drafts <= 0:
            failures.append(f"dp_rank={dp_rank}: no correct drafts")
        else:
            total_correct_drafts += correct_drafts

    return {
        "score": total_correct_drafts,
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
    "DP_SIZE": str(DP_SIZE),
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
    "test_name": "gpt-oss-20b-spec-dpa",
    "variant": "TP8+EP8+DP2+EAGLE3",
    "launch": {
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-20b-spec-dpa-e2e",
    },
    "gates": [
        {
            "dataset": "speculative_decoding_dpa",
            "label": "speculative-decoding-dpa",
            "runner": spec_dpa_runner,
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
