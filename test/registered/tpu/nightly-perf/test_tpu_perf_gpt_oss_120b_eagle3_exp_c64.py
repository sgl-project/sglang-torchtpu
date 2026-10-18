"""Throughput sweep for gpt-oss-120b + EAGLE3 at c64 on the experimental spec worker.

Same launch, workload and gate as test_tpu_perf_gpt_oss_120b_eagle3_c64.py, with
SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER=1 routing EAGLE3 to TpuEagle3Worker;
reading the two side by side compares the workers.

The rest of this docstring describes the default-worker file it copies.

Read against test_tpu_perf_gpt_oss_120b_c64.py: same model, same workload, same
concurrency, the draft model being the only difference. That pairing is the
point -- speculative decoding trades extra compute per step for fewer steps, and
whether that is a win at 64-wide batching is exactly what a throughput trend can
answer and a single-request test cannot.

See test_tpu_perf_gpt_oss_120b.py for the ADDING A MODEL guide and for why perf
serves from its own recipe rather than the accuracy nightly's.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

# Longer than the non-speculative c64 file: bring-up also loads the draft model
# and compiles the draft forward. Drives the JobSet poll budget.
register_tpu_ci(
    est_time=3000,
    suite="nightly-tpu-perf-gpt-oss-120b-exp-c64",
    nightly=True,
    labels=[
        "singlehost",
        "gpt-oss-120b",
        "perf",
        "spec",
        "eagle3",
        "experimental-spec",
    ],
)

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

WORKLOAD = "8k1k"
ISL, OSL, CONCUR = 8192, 1024, 64

MODEL = "openai/gpt-oss-120b"
WEIGHTS = "/weights/gpt-oss-120b"
DRAFT_MODEL = "lmsys/EAGLE3-gpt-oss-120b-bf16"

# Three drafts per verify, matching the DFlash sibling's block size so the two
# legs propose the same amount of work. topk=1 makes this a chain, and the arg
# hook then resolves num_draft_tokens to num_steps + 1 (3 drafts + 1 bonus).
NUM_STEPS, EAGLE_TOPK = 3, 1

# The sweep drives random-ids prompts, which have no structure for a drafter to
# predict: real acceptance would collapse toward 1 and the throughput would not
# describe any served workload. Pin it instead, near what GSM8K actually
# measures, so the number is representative and comparable across drafters.
# Fractional: the default match-expected method commits 2 or 3 per verify so the
# mean lands on 2.5.
# Output text is meaningless under this knob -- never set it on an accuracy gate.
SIMULATE_ACC_LEN = 2.5


MODEL_CONFIG = {
    "model": MODEL,
    "test_name": "gpt-oss-120b-eagle3-exp-c64",
    "id": "perf-gpt-oss-120b-eagle3-exp-c64",
    "display_name": "GPT-OSS 120B EAGLE3 experimental worker (c64)",
    "variant": "TP8+EP8+EAGLE3-exp-c64",
    "launch": {
        # Deltas over the c64 baseline: the draft model and its chain. The draft
        # weights come out of the same MEM_FRACTION_STATIC budget as the target's
        # KV pool, so if bring-up OOMs, that fraction is the knob -- not the pool
        # size, which has to stay at the baseline's for the numbers to compare.
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "env": {
            "MODEL": WEIGHTS,
            "ISL": str(ISL),
            "OSL": str(OSL),
            "CONCUR": str(CONCUR),
            "DISABLE_RADIX_CACHE": "1",
            "TPU_COMPILE_TOKENS": str(ISL),
            "TPU_COMPILE_BS": str(CONCUR),
            "ENABLE_PROFILER_ENDPOINT": "0",
            "SGLANG_SIMULATE_ACC_LEN": str(SIMULATE_ACC_LEN),
            "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER": "1",
            "EXTRA_ARGS": (
                f"--served-model-name {MODEL} "
                "--speculative-algorithm EAGLE3 "
                f"--speculative-draft-model-path {DRAFT_MODEL} "
                f"--speculative-num-steps {NUM_STEPS} "
                f"--speculative-eagle-topk {EAGLE_TOPK}"
            ),
        },
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "perf-gpt-oss-eagle3-exp-c64",
    },
    "gates": [
        {
            "dataset": "bench_serving",
            "label": WORKLOAD,
            "sweep": [
                {
                    "workload": WORKLOAD,
                    "isl": ISL,
                    "osl": OSL,
                    "concurrency": CONCUR,
                }
            ],
            "bench_env": {
                "DATASET_NAME": "random-ids",
                "TOKENIZE_PROMPT": "1",
                "MODEL": WEIGHTS,
                "SERVED_MODEL_NAME": MODEL,
            },
        }
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
