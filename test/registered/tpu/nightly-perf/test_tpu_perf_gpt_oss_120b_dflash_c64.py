"""Throughput sweep for gpt-oss-120b + DFlash on the 8k1k workload at c64.

The third leg of the c64 set: read against test_tpu_perf_gpt_oss_120b_c64.py for
the cost of speculation at all, and against the EAGLE3 sibling for which drafter
wins at 64-wide batching. All three serve the same model on the same workload at
the same concurrency, so the only free variable is the draft path.

See test_tpu_perf_gpt_oss_120b.py for the ADDING A MODEL guide.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

# Longer than the non-speculative c64 file: bring-up also loads the draft model
# and compiles the draft forward. Drives the JobSet poll budget.
register_tpu_ci(
    est_time=3000,
    suite="nightly-tpu-perf-gpt-oss-120b-dflash-c64",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "perf", "spec", "dflash"],
)

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

WORKLOAD = "8k1k"
ISL, OSL, CONCUR = 8192, 1024, 64

MODEL = "openai/gpt-oss-120b"
WEIGHTS = "/weights/gpt-oss-120b"
DRAFT_MODEL = "z-lab/gpt-oss-120b-DFlash"

# Fixed by the z-lab DFlash checkpoint; the arg hook pins num_steps and
# eagle_topk to 1, so block size is the only speculative knob that applies.
DFLASH_BLOCK_SIZE = 3

# The sweep drives random-ids prompts, which have no structure for a drafter to
# predict: real acceptance would collapse toward 1 and the throughput would not
# describe any served workload. Pin it instead, near what GSM8K actually
# measures, so the number is representative and comparable across drafters.
# Fractional: the default match-expected method commits 2 or 3 per verify so the
# mean lands on 2.5.
# Output text is meaningless under this knob -- never set it on an accuracy gate.
SIMULATE_ACC_LEN = 2.5

# DFlash is the only algorithm whose scheduler auto-enables min-free-slots-delay
# (4 slots at max-running 64). It holds decode rows empty that the TPU batch
# bucket pays for regardless, and the prefill batching it buys is worth nothing
# at ISL 8192 against an 8192 token bucket. 1 disables.
MIN_FREE_SLOTS_DELAY = 1


MODEL_CONFIG = {
    "model": MODEL,
    "test_name": "gpt-oss-120b-dflash-c64",
    "id": "perf-gpt-oss-120b-dflash-c64",
    "display_name": "GPT-OSS 120B DFlash (c64)",
    "variant": "TP8+EP8+DFLASH-c64",
    "launch": {
        # Deltas over the c64 baseline: the draft model and its block size. As
        # with EAGLE3, the draft weights share MEM_FRACTION_STATIC with the
        # target's KV pool -- tune that fraction on an OOM, not the pool size,
        # which has to match the baseline for the numbers to compare.
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
            # Match the DFlash accuracy gates, so this measures the configuration
            # they validate. Only the sites that can differ across ranks are
            # broadcast; the deterministic ones would cost a collective per step.
            "SGLANG_SPEC_TP_SYNC": "rng",
            "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER": "1",
            "EXTRA_ARGS": (
                f"--served-model-name {MODEL} "
                "--speculative-algorithm DFLASH "
                f"--speculative-draft-model-path {DRAFT_MODEL} "
                f"--speculative-dflash-block-size {DFLASH_BLOCK_SIZE} "
                f"--min-free-slots-delay {MIN_FREE_SLOTS_DELAY}"
            ),
        },
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "perf-gpt-oss-dflash-c64",
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
