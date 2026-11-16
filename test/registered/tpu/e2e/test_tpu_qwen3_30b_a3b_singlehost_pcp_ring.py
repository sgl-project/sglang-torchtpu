"""Qwen3-30B-A3B-FP8 single-host ring-mode prefill context parallelism nightly.

PCP8 with ``SGLANG_TPU_PCP_MODE=ring``: the KV cache is page-striped across the
eight cores (DCP8 layout), zigzag prefills attend the cached prefix through the
fused ring kernel, and decode plus short extends run the DCP path over the same
stripes. Same gates as the gather-mode test: a greedy needle probe over prompts
landing in the smallest and in a larger local token bucket, alone and in shared
context-parallel batches (with short prompts and prompts that come back with a
cached prefix, which is what sends keys around the ring), and few-shot GSM8K.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.needle_probe import needle_runner
from sglang.test.tpu import model_config_main
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

# Nightly only: the ring kernel needs all eight cores of the host in one mesh,
# and the 30B FP8 checkpoint is the configuration ring mode is validated for.
register_tpu_ci(
    est_time=1200,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "qwen3-30b-a3b", "pcp-ring"],
)

SERVED_MODEL_NAME = "qwen3-30b-a3b"
MODEL_PATH = resolve_tpu_model_path("Qwen/Qwen3-30B-A3B-FP8")
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))
# The gather-mode floor (PCP8 + MoE DP8 scored 0.91 on these 200 questions);
# ring mode computes the same attention, so it has to clear the same bar.
GSM8K_ACCURACY = 0.86

LAUNCH_ENV = {
    "MODEL": MODEL_PATH,
    # Ring-mode CP batches reserve two page-aligned blocks per request on every
    # core, so their local token buckets are multiples of 2 x page size (512):
    # 512 and 1024 hold the needles, 2048 a full chunk; 8192 covers the striped
    # (non-CP) extends.
    "TPU_COMPILE_TOKENS": "512 1024 2048 8192",
    "TPU_COMPILE_BS": "8 16 32",
    "MAX_RUNNING": "32",
    "TORCH_COMPILE_MAX_BS": "32",
    "ENABLE_PROFILER_ENDPOINT": "0",
    "DISABLE_RADIX_CACHE": "0",
    "EXTRA_ARGS": f"--served-model-name {SERVED_MODEL_NAME}",
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "qwen3-30b-a3b-pcp-ring",
    "variant": "PCP8-ring+DCP8+MoE-DP8",
    "launch": {
        "recipe": "recipe/qwen3_30b_a3b_fp8_pcp_ring.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "qwen3-30b-a3b-pcp-ring-e2e",
    },
    "gates": [
        {
            "dataset": "needle",
            "label": "needle-recall",
            "runner": needle_runner,
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
