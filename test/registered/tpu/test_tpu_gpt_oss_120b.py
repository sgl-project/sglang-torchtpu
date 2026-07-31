"""gpt-oss-120b single-host e2e accuracy: GPQA Diamond, MMLU and AIME 2025.

Three gates against one server boot, all anchored on the model card's low-effort,
no-tools column (Table 3, arXiv:2508.10925): GPQA 0.671, MMLU 0.859, AIME25 0.504.

Both qualifiers are load-bearing. The card gives a different number per effort level
(GPQA 0.671 / 0.731 / 0.801) and the with-tools column is higher again (AIME25 0.729
vs 0.504) -- this server executes no tools.

GPQA uses OpenAI's own gpt_oss.evals because 0.671 is only reproducible with their
permuted-choice prompt and grader; MMLU and AIME use run_eval, itself adapted from
OpenAI's simple-evals.

All three gate at atol 0.03, capped by the tolerance contract. Headroom differs by
sample count: MMLU ~3.7 sigma, GPQA ~1.8, AIME only ~0.9 -- 30 questions at p=0.5 is
the noisiest case there is, so AIME trips ~18% of runs on sampling noise alone.

Serving config lives in recipe/gpt_oss_120b.sh, which CI runs verbatim. Weights come
from the shared ROX PVC model-weights-rox at /weights
(pvc-model-weights-rox.yaml.tmpl); gpt-oss itself is runtime-installed --no-deps so
its pins never touch the pinned torch.
"""

import os
import subprocess
import sys

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import dump_metric
from sglang.test.tpu import model_config_main

# est_time drives the JobSet deadline, so an under-estimate kills the run. ~3000
# generations on one boot (GPQA 4x198, MMLU 2000, AIME 8x30), much of it low-effort
# reasoning up to 8k tokens. weekly-update-est-time.yml refreshes it from real p90s.
register_tpu_ci(est_time=10800, suite="nightly-tpu-4-v7", nightly=True)

MODEL_PATH = "/weights/gpt-oss-120b"
# Must match --served-model-name in recipe/gpt_oss_120b.sh: the sampler below sends
# this as the model id and the request 404s if the two disagree.
SERVED_MODEL_NAME = "gpt-oss-120b"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))
REASONING_EFFORT = "low"

# --- model card ground truths (Table 3; low effort, no tools) ---
GPQA_GROUND_TRUTH = 0.671
MMLU_GROUND_TRUTH = 0.859
AIME25_GROUND_TRUTH = 0.504

# 198 questions -> single-pass sigma 3.3pts, under 1 sigma for a 3pt gate. 4 repeats
# put the mean's SE at ~1.7pts, making the floor ~1.8 sigma. The card uses 8; 4 halves
# CI time. Do NOT go below 4.
GPQA_ATOL = 0.03
GPQA_N_REPEATS = 4

# 2000 questions -> sigma ~0.8pts, so 3pts is ~3.7 sigma. The headroom also absorbs
# prompt-formatting differences between simple-evals and OpenAI's internal harness.
MMLU_ATOL = 0.03
MMLU_NUM_EXAMPLES = 2000

# The noisiest gate: 30 questions (AIME2025-I + II) at p=0.504, the worst case for
# binomial noise since p(1-p) peaks at 0.5. Single-pass sigma 9.1pts; at 8 repeats the
# mean's SE is 3.2pts, leaving the 3pt floor 0.9 sigma below the reference, so it trips
# ~18% of runs on noise alone. 2 sigma would take 37 repeats, 3 sigma 83. atol is
# capped at 3pts by contract, so repeats are the only lever -- re-run before
# investigating a red result.
AIME25_ATOL = 0.03
AIME25_REPEAT = 8


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
        max_tokens=8192,
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


MODEL_CONFIG = {
    "model": MODEL_PATH,
    "test_name": "gpt-oss-120b",
    "variant": "TP8+EP8",
    "launch": {
        "recipe": "recipe/gpt_oss_120b.sh",
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
            "max_tokens": 8192,
        },
        {
            "dataset": "aime25",
            "label": "aime25_low",
            "baseline": AIME25_GROUND_TRUTH,
            "atol": AIME25_ATOL,
            "num_threads": 32,
            "repeat": AIME25_REPEAT,
            "reasoning_effort": REASONING_EFFORT,
            "max_tokens": 32768,
        },
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
