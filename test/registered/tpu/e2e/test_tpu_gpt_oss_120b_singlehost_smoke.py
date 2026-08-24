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
count: MMLU ~3.7 sigma, GPQA ~2.6 at the card's 8 repeats.

Measured 2026-08-04 on one v7x host at this config: GPQA 0.6553, MMLU 0.8670. MMLU is
sensitive to answer extraction -- this model ends its replies "Answer: $B$" -- so check
the unextractable rate before suspecting the serving stack.

Serving config is recipe/gpt_oss_120b/server.sh, the one script for this model and what
the perf sweep runs too. LAUNCH_ENV below is the eval delta on top of it, itemised
there; the launcher logs it on boot, so a red nightly reproduces as that line plus
./server.sh. Weights come from the shared ROX PVC model-weights-rox at /weights
(pvc-model-weights-rox.yaml.tmpl); gpt-oss itself is runtime-installed --no-deps so its
pins never touch the pinned torch.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import gpqa_openai_runner, model_config_main

# est_time drives the JobSet poll budget, so an under-estimate kills the run. Measured
# 1913s / 1944s on the 2026-08-05 and 08-04 nightlies; 2400 leaves headroom for GPQA's
# temperature-1.0 variance. Not auto-refreshed: update_est_time.py omits the tpu backend.
register_tpu_ci(est_time=2400, suite="nightly-tpu-4-v7", nightly=True)

# Must match --served-model-name in LAUNCH_ENV: every gate sends this as the model id,
# and it is what the server echoes back in its responses.
SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))
REASONING_EFFORT = "low"

# --- model card ground truths (Table 3; low effort, no tools) ---
GPQA_GROUND_TRUTH = 0.671
MMLU_GROUND_TRUTH = 0.859

# 198 questions -> single-pass sigma 3.3pts, under 1 sigma for a 3pt gate, so the gate
# is only meaningful on a mean of several passes. 8 is the card's own repeat count: the
# mean's SE is 3.3/sqrt(8) = 1.2pts, putting the floor ~2.6 sigma below the reference.
# Do NOT lower it -- at 4 the floor is ~1.8 sigma and the gate reds on noise alone.
GPQA_ATOL = 0.03
GPQA_N_REPEATS = 8

# 2000 questions -> sigma ~0.8pts, so 3pts is ~3.7 sigma. The headroom also absorbs
# prompt-formatting differences between simple-evals and OpenAI's internal harness.
MMLU_ATOL = 0.03
MMLU_NUM_EXAMPLES = 2000

# Completion budget both gates are sized against: low-effort generations finish well
# under it, and CONTEXT_LEN leaves room for it on top of the longest prompt.
MAX_TOKENS = 8192


# server.sh owns how the model is served -- tp8+ep8, MXFP4 auto-detect, RPA v3 (the only
# backend implementing attention sinks), SC gather off, mem_fraction 0.6 -- but its
# defaults size it for the throughput sweep. Everything below restates the accuracy
# nightly's configuration, so the server this file launches matches what the GPQA-only
# nightly launched before these gates shared a boot:
#   MODEL           the ROX PVC mount; server.sh defaults to the HF id.
#   EXTRA_ARGS      Harmony's analysis channel -> reasoning_content, so a harness grades
#                   final-channel text only. served-model-name is SERVED_MODEL_NAME.
#   KV_CACHE_DTYPE  empty = bf16 KV, what GPQA 0.671 was validated on (2026-07-28);
#                   server.sh defaults to fp8_e4m3, which nothing here is anchored to.
#   PAGE_SIZE       128, the platform default the nightly ran on. server.sh drops to 64
#                   to raise the prefix-cache hit rate, which no gate here depends on.
#   DISABLE_RADIX_CACHE  prefix caching off, as the nightly ran it. server.sh turns it
#                   on for the cache-hit sweep; leaving it on would let a repeated
#                   prompt skip a real prefill.
#   MAX_TOTAL_TOKENS  empty = let sglang size the KV pool from mem_fraction, as the
#                   nightly did. server.sh's 2*CONCUR*CONTEXT_LEN is sized for its fp8
#                   pool; at bf16 that formula would ask for most of the budget.
#   ISL/OSL/CONCUR  server.sh derives the prefill chunk from ISL, and the bs ceiling
#                   from CONCUR. CONCUR=32 reproduces the nightly's MAX_RUNNING and
#                   TORCH_COMPILE_MAX_BS.
#   CONTEXT_LEN     longest GPQA prompt (~2.9k tokens) + the MAX_TOKENS budget. The
#                   sweep's default is ISL+OSL+64 = 2112, which a GPQA prompt overruns.
#   TPU_COMPILE_*   one prefill bucket (= the 4096 chunk ISL derives) and one decode
#                   bucket, which keeps bring-up ~10 min; the derived default is powers
#                   of two up to CONCUR, i.e. six decode compiles.
#   ENABLE_PROFILER_ENDPOINT  launch_server.sh defaults it to 1; a nightly wants none.
#
# With these, the argv launch_server.sh builds is identical to the GPQA-only nightly's.
LAUNCH_ENV = {
    "MODEL": "/weights/gpt-oss-120b",
    "EXTRA_ARGS": f"--reasoning-parser gpt-oss --served-model-name {SERVED_MODEL_NAME}",
    "KV_CACHE_DTYPE": "",
    # Was "" so server.sh's `${MAX_TOTAL_TOKENS-...}` passed through empty and let
    # sglang size the pool from mem_fraction. That was written for bf16 KV, where
    # the recipe's formula would have claimed most of the budget. The KV cache is
    # fp8_e4m3 now (see the pool log), so the same mem_fraction buys ~25x more
    # tokens than the run can use: 13,183,232 tokens / 113 GB of pool, leaving the
    # XLA compile with 142.23 G of HLO temporaries against 94.74 G of HBM. The
    # server is SIGKILLed 92s in, so the gate reports neither gpqa nor mmlu.
    #
    # 2 * CONCUR * CONTEXT_LEN is the recipe's own formula and covers every
    # resident prefix at this shape (32 x 16384 in flight). Capping the pool
    # changes capacity, not arithmetic -- no request is truncated or rejected at
    # this concurrency, so the scores are unaffected.
    "MAX_TOTAL_TOKENS": str(2 * 32 * 16384),
    "PAGE_SIZE": "128",
    "DISABLE_RADIX_CACHE": "1",
    "MEM_FRACTION_STATIC": "0.60",
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
            "runner": gpqa_openai_runner(
                model=SERVED_MODEL_NAME,
                n_repeats=GPQA_N_REPEATS,
                reasoning_effort=REASONING_EFFORT,
                max_tokens=MAX_TOKENS,
            ),
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
