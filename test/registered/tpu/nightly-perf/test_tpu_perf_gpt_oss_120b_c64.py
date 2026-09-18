"""Throughput sweep for gpt-oss-120b on the 8k1k workload at concurrency 64.

The c16 sibling (test_tpu_perf_gpt_oss_120b.py) carries the ADDING A MODEL guide
and the rationale for serving perf from its own recipe; read it first. This file
is the same measurement one concurrency step up, and the baseline the two
speculative-decoding perf files are read against.

WHY A SEPARATE FILE, not a second sweep point: CONCUR sizes the server, not just
the client -- max-running-requests, the compile buckets and the KV pool all
derive from it. Measuring c16 on a server built for 64 would answer a different
question than the c16 file does, so the trends would not be comparable.
"""

import os
import sys
from copy import deepcopy
from pathlib import Path

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmark.tpu.recipe.configs.gpt_oss_120b.server import CONFIG

# One bring-up plus one sweep leg. Longer than the c16 file's 1800: a 64-wide
# decode bucket compiles slower and the KV pool is 4x. Drives the JobSet poll
# budget, so an under-estimate kills the run.
register_tpu_ci(
    est_time=2400,
    suite="nightly-tpu-perf-gpt-oss-120b-c64",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "perf"],
)

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))

WORKLOAD = "8k1k"
ISL, OSL, CONCUR = 8192, 1024, 64

MODEL = "openai/gpt-oss-120b"
WEIGHTS = "/weights/gpt-oss-120b"


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
    }
)

MODEL_CONFIG = {
    "model": MODEL,
    "test_name": "gpt-oss-120b-c64",
    "id": "perf-gpt-oss-120b-c64",
    "display_name": "GPT-OSS 120B (c64)",
    "variant": "TP8+EP8-c64",
    "launch": {
        **LAUNCH_CONFIG,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "perf-gpt-oss-c64",
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
