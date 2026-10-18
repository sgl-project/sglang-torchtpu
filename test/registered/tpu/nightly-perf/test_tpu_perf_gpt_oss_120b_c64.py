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

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.utils.accuracy_base import recipe_server_args

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
CONFIG_MODULE = "recipe/configs/gpt_oss_120b/perf_c64.py"
ISL, OSL = 8192, 1024
# The server is sized for exactly this concurrency, so drive what it was
# sized for: the recipe's profile is the one source of the number.
CONCUR = recipe_server_args(CONFIG_MODULE)["max_running_requests"]

MODEL = "openai/gpt-oss-120b"
WEIGHTS = "/weights/gpt-oss-120b"


MODEL_CONFIG = {
    "model": MODEL,
    "test_name": "gpt-oss-120b-c64",
    "id": "perf-gpt-oss-120b-c64",
    "display_name": "GPT-OSS 120B (c64)",
    "variant": "TP8+EP8-c64",
    "launch": {
        # Same deltas as the c16 file; see it for why each one is here. The only
        # change is CONCUR, from which server.sh derives MAX_RUNNING,
        # TORCH_COMPILE_MAX_BS and a 2*CONCUR*CONTEXT_LEN = 1.19M-token fp8 KV
        # pool -- still under the ~1.5M where its extend precompile OOMs, but
        # with far less slack than c16's 297k.
        "config": CONFIG_MODULE,
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
