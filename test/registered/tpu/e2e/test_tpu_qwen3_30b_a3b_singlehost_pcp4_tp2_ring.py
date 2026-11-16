"""Qwen3-30B-A3B-FP8 single-host ring-mode PCP4 x attention TP 2 nightly.

``SGLANG_TPU_PCP_MODE=ring`` with ``--attn-cp-size 4`` on tp 8: each core keeps
2 of the 4 KV heads (attention TP 2; an fp8 pool packs 4 rows per word, so one
head per core would waste half of it), the KV cache is page-striped over each
attention-CP group (ranks ``t, t + 2, t + 4, t + 6``), zigzag prefills attend
the cached prefix through the fused ring kernel on a ``(4, 2)`` mesh ringing
along CP, and decode plus short extends run the DCP path over the same stripes,
merging over the attention-CP group. The MoE runs MoE DP 4 x EP 2 (the
recipe's default at ATTN_CP_SIZE 4). Same gates as the gather-mode test: a greedy needle probe over prompts
landing in the smallest and in a larger local token bucket, alone and in shared
context-parallel batches (with short prompts and prompts that come back with a
cached prefix, which is what sends keys around the ring), and few-shot GSM8K.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.needle_probe import needle_runner
from sglang.test.tpu import model_config_main
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

# Nightly only: the ring kernel needs all eight cores of the host in one mesh.
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
    # 512 and 1024 hold the needles, 2048 a full 8k chunk over 4 CP ranks; 8192
    # covers the striped (non-CP) extends.
    "TPU_COMPILE_TOKENS": "512 1024 2048 8192",
    "TPU_COMPILE_BS": "8 16 32",
    "ATTN_CP_SIZE": "4",
    "MAX_RUNNING": "32",
    "TORCH_COMPILE_MAX_BS": "32",
    "ENABLE_PROFILER_ENDPOINT": "0",
    "DISABLE_RADIX_CACHE": "0",
    "EXTRA_ARGS": f"--served-model-name {SERVED_MODEL_NAME}",
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "qwen3-30b-a3b-pcp4-tp2-ring",
    "variant": "PCP4xTP2-ring+DCP4+MoE-DP4xEP2",
    "launch": {
        "recipe": "recipe/qwen3_30b_a3b_fp8_pcp_ring.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "qwen3-30b-a3b-pcp4-tp2-ring-e2e",
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
