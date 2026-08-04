"""gpt-oss-120b single-host e2e accuracy: GPQA Diamond and MMLU.

Two gates against one server boot, both anchored on the model card's low-effort,
no-tools column (Table 3, arXiv:2508.10925): GPQA 0.671, MMLU 0.859.

Both qualifiers are load-bearing. The card gives a different number per effort level
(GPQA 0.671 / 0.731 / 0.801) and the with-tools column is higher again -- this server
executes no tools.

GPQA uses OpenAI's own gpt_oss.evals because 0.671 is only reproducible with their
permuted-choice prompt and grader; MMLU uses run_eval, itself adapted from OpenAI's
simple-evals.

Both gate at atol 0.03, capped by the tolerance contract. Headroom differs by sample
count: MMLU ~3.7 sigma, GPQA ~1.8.

Serving config is recipe/gpt_oss_120b/server.sh, the one script for this model and what
the perf sweep runs too. LAUNCH_ENV below is the eval delta on top of it, itemised
there; the launcher logs it on boot, so a red nightly reproduces as that line plus
./server.sh. Weights come from the shared ROX PVC model-weights-rox at /weights
(pvc-model-weights-rox.yaml.tmpl); gpt-oss itself is runtime-installed --no-deps so its
pins never touch the pinned torch.
"""

import os
import subprocess
import sys

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import dump_metric
from sglang.test.tpu import model_config_main

# est_time drives the JobSet deadline, so an under-estimate kills the run. ~2800
# generations on one boot (GPQA 4x198, MMLU 2000), much of it low-effort reasoning up
# to 8k tokens. weekly-update-est-time.yml refreshes it from real p90s.
register_tpu_ci(est_time=10800, suite="nightly-tpu-4-v7", nightly=True)

# Must match --served-model-name in LAUNCH_ENV: every gate sends this as the model id,
# and it is what the server echoes back in its responses.
SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))
REASONING_EFFORT = "low"

# --- model card ground truths (Table 3; low effort, no tools) ---
GPQA_GROUND_TRUTH = 0.671
MMLU_GROUND_TRUTH = 0.859

# 198 questions -> single-pass sigma 3.3pts, under 1 sigma for a 3pt gate. 4 repeats
# put the mean's SE at ~1.7pts, making the floor ~1.8 sigma. The card uses 8; 4 halves
# CI time. Do NOT go below 4.
GPQA_ATOL = 0.03
GPQA_N_REPEATS = 4

# 2000 questions -> sigma ~0.8pts, so 3pts is ~3.7 sigma. The headroom also absorbs
# prompt-formatting differences between simple-evals and OpenAI's internal harness.
MMLU_ATOL = 0.03
MMLU_NUM_EXAMPLES = 2000

# Completion budget both gates are sized against: low-effort generations finish well
# under it, and CONTEXT_LEN leaves room for it on top of the longest prompt.
MAX_TOKENS = 8192


def _gpqa_via_openai_harness(base_url: str) -> dict:
    """Score GPQA Diamond with OpenAI's own gpt_oss.evals harness."""
    # Pure-python eval code; --no-deps so its pins never touch the pinned
    # TPU torch stack. openai/pandas/blobfile are image deps (test extra).
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "gpt-oss"],
        check=True,
    )
    # The openai client insists on a key even for a local base_url.
    os.environ.setdefault("OPENAI_API_KEY", "dummy")

    from gpt_oss.evals.chat_completions_sampler import ChatCompletionsSampler
    from gpt_oss.evals.gpqa_eval import GPQAEval

    sampler = ChatCompletionsSampler(
        model=SERVED_MODEL_NAME,
        reasoning_model=True,
        reasoning_effort=REASONING_EFFORT,
        temperature=1.0,
        base_url=base_url + "/v1",
        # Their CLI uses 131072; capped to fit CONTEXT_LEN. Proven
        # non-truncating: low-effort generations complete well under 8k.
        max_tokens=MAX_TOKENS,
    )
    result = GPQAEval(n_repeats=GPQA_N_REPEATS, n_threads=32)(sampler)
    print(
        f"gpqa_diamond (openai harness, n_repeats={GPQA_N_REPEATS}): "
        f"score={result.score:.4f} metrics={result.metrics}"
    )
    # Built-in backends dump their own metrics; a runner must do its own.
    dump_metric(
        "gpqa_diamond_openai_harness",
        result.score,
        labels={"model": "openai/gpt-oss-120b", "eval": "gpt_oss.evals"},
    )
    return {"score": result.score}


# server.sh owns how the model is served -- tp8+ep8, MXFP4 auto-detect, RPA v3 (the only
# backend implementing attention sinks), SC gather off, mem_fraction 0.6, page 64 -- and
# is sized by default for the throughput sweep. The gates need seven values changed:
#   MODEL           the ROX PVC mount; server.sh defaults to the HF id.
#   EXTRA_ARGS      Harmony's analysis channel -> reasoning_content, so a harness grades
#                   final-channel text only. served-model-name is SERVED_MODEL_NAME.
#   KV_CACHE_DTYPE  empty = bf16 KV, what GPQA 0.671 was validated on (2026-07-28);
#                   server.sh defaults to fp8_e4m3, which nothing here is anchored to.
#   ISL/OSL/CONCUR  server.sh derives the prefill chunk from ISL, and the KV pool plus
#                   the bs ceiling from CONCUR. CONCUR=32 holds MAX_TOTAL_TOKENS
#                   (2*CONCUR*CONTEXT_LEN = 1.05M) under the ~1.5M at which the extend
#                   precompile OOMs, and matches MAX_RUNNING to the decode bucket below.
#   CONTEXT_LEN     longest GPQA prompt (~2.9k tokens) + the MAX_TOKENS budget. The
#                   sweep's default is ISL+OSL+64 = 2112, which a GPQA prompt overruns.
#   TPU_COMPILE_*   one prefill bucket (= the 4096 chunk ISL derives) and one decode
#                   bucket, which keeps bring-up ~10 min; the derived default is powers
#                   of two up to CONCUR, i.e. six decode compiles.
#   ENABLE_PROFILER_ENDPOINT  launch_server.sh defaults it to 1; a nightly wants none.
LAUNCH_ENV = {
    "MODEL": "/weights/gpt-oss-120b",
    "EXTRA_ARGS": f"--reasoning-parser gpt-oss --served-model-name {SERVED_MODEL_NAME}",
    "KV_CACHE_DTYPE": "",
    "ISL": "4096",
    "OSL": str(MAX_TOKENS),
    "CONCUR": "32",
    "CONTEXT_LEN": "16384",
    "TPU_COMPILE_TOKENS": "4096",
    "TPU_COMPILE_BS": "32",
    "ENABLE_PROFILER_ENDPOINT": "0",
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-120b",
    "variant": "TP8+EP8",
    "launch": {
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpqa-e2e",
    },
    "gates": [
        {
            "dataset": "gpqa",
            "label": "gpqa_diamond_low",
            "runner": _gpqa_via_openai_harness,
            "baseline": GPQA_GROUND_TRUTH,
            "atol": GPQA_ATOL,
        },
        {
            "dataset": "mmlu",
            "label": "mmlu_low",
            "baseline": MMLU_GROUND_TRUTH,
            "atol": MMLU_ATOL,
            "num_examples": MMLU_NUM_EXAMPLES,
            "num_threads": 32,
            "reasoning_effort": REASONING_EFFORT,
            "max_tokens": MAX_TOKENS,
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
