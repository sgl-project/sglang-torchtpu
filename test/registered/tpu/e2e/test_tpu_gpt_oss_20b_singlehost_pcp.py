"""GPT-OSS-20B single-host prefill context parallelism PR gate.

PCP8 with MoE DP8 on the batched RPA backend, radix cache on. A greedy needle
probe checks that prompts landing in the smallest and in a larger local token
bucket, alone and in shared context-parallel batches, keep every code word. The
shared batches also carry prompts shorter than the zigzag split and the two
prompts sent first, which come back with a cached prefix. Few-shot GSM8K, whose
prompts share their examples, checks answers end to end.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.needle_probe import needle_runner
from sglang.test.tpu import model_config_main
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

# PCP is an attention feature; the checkpoint only has to be good enough for the
# probe's assertions. GPT-OSS-20B loads in a fraction of the 30B's time, which is
# what made test_tpu_qwen3_30b_a3b_singlehost_pcp.py the most expensive launch in
# the e2e gate. That one now runs nightly and keeps the 30B FP8 configuration.
register_tpu_ci(
    est_time=600,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "gpt-oss-20b", "pcp"],
)

SERVED_MODEL_NAME = "gpt-oss-20b"
MODEL_PATH = resolve_tpu_model_path("openai/gpt-oss-20b")
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))
# The floor the other GPT-OSS-20B gates use; this model scores ~0.775 on GSM8K
# where Qwen3-30B-A3B scored 0.915, so the 0.86 floor of the nightly PCP test
# does not transfer.
GSM8K_ACCURACY = 0.60

LAUNCH_ENV = {
    "MODEL": MODEL_PATH,
    # CP batches bucket their local rows (1/8 of the tokens) here too: 512 and
    # 1024 hold the needles, 2048 a full chunk; 8192 covers non-CP batches.
    "TPU_COMPILE_TOKENS": "512 1024 2048 8192",
    "TPU_COMPILE_BS": "8 16 32",
    "MAX_RUNNING": "32",
    "TORCH_COMPILE_MAX_BS": "32",
    "ENABLE_PROFILER_ENDPOINT": "0",
    "DISABLE_RADIX_CACHE": "0",
    "EXTRA_ARGS": f"--reasoning-parser gpt-oss --served-model-name {SERVED_MODEL_NAME}",
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-20b-pcp",
    "variant": "PCP8+MoE-DP8",
    "launch": {
        "recipe": "recipe/gpt_oss_20b_pcp.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-20b-pcp-e2e",
    },
    "gates": [
        {
            "dataset": "needle",
            "label": "needle-recall",
            # GPT-OSS can open with reasoning text, so the answer needs
            # more room than the 12 tokens the Qwen checkpoint needed.
            "runner": lambda base_url: needle_runner(base_url, max_new_tokens=48),
        },
        {
            "dataset": "gsm8k",
            "label": "gsm8k",
            "baseline": GSM8K_ACCURACY,
            "num_examples": 200,
            "num_threads": 32,
            "max_tokens": 512,
            "temperature": 0.0,
            # Few-shot completion prompts: the checkpoint's chat template would
            # spend the token budget on thinking.
            "api": "completion",
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
