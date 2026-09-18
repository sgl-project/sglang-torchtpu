"""Throughput sweep for gpt-oss-120b + EAGLE3 on the 8k1k workload at c64.

Read against test_tpu_perf_gpt_oss_120b_c64.py: same model, same workload, same
concurrency, the draft model being the only difference. That pairing is the
point -- speculative decoding trades extra compute per step for fewer steps, and
whether that is a win at 64-wide batching is exactly what a throughput trend can
answer and a single-request test cannot.

See test_tpu_perf_gpt_oss_120b.py for the ADDING A MODEL guide and for why perf
serves from its own recipe rather than the accuracy nightly's.
"""

import os
import sys
from copy import deepcopy
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmark.tpu.recipe.configs.gpt_oss_120b.server import CONFIG

# Longer than the non-speculative c64 file: bring-up also loads the draft model
# and compiles the draft forward. Drives the JobSet poll budget.
register_tpu_ci(
    est_time=3000,
    suite="nightly-tpu-perf-gpt-oss-120b-eagle3-c64",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "perf", "spec", "eagle3"],
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


LAUNCH_CONFIG = deepcopy(CONFIG)
LAUNCH_CONFIG["server_args"].update(
    {
        "context_length": int(ISL) + int(OSL) + 64,
        "max_running_requests": int(CONCUR),
        "torch_compile_max_bs": int(CONCUR),
        "chunked_prefill_size": min(max(int(ISL), 2048), 8192),
        "max_prefill_tokens": min(max(int(ISL), 2048), 8192),
        "max_total_tokens": 2 * (int(CONCUR)) * (int(ISL) + int(OSL) + 64),
        "model_path": WEIGHTS,
        "disable_radix_cache": True,
        "tpu_compile_tokens": [ISL],
        "tpu_compile_bs": [CONCUR],
        "served_model_name": MODEL,
        "speculative_algorithm": "EAGLE3",
        "speculative_draft_model_path": DRAFT_MODEL,
        "speculative_num_steps": NUM_STEPS,
        "speculative_eagle_topk": EAGLE_TOPK,
    }
)
LAUNCH_CONFIG["runtime_env"].update(
    {
        "SGLANG_SIMULATE_ACC_LEN": str(SIMULATE_ACC_LEN),
    }
)

MODEL_CONFIG = {
    "model": MODEL,
    "test_name": "gpt-oss-120b-eagle3-c64",
    "id": "perf-gpt-oss-120b-eagle3-c64",
    "display_name": "GPT-OSS 120B EAGLE3 (c64)",
    "variant": "TP8+EP8+EAGLE3-c64",
    "launch": {
        **LAUNCH_CONFIG,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "perf-gpt-oss-eagle3-c64",
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
