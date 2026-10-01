"""Throughput sweep for Qwen3.5-397B-A17B on the 8k1k workload, tp16 / 2 hosts.

The single-host siblings in this directory launch their own server through
TPULauncherTestCase. This one cannot: 397B of bf16 weights needs both hosts of a
2x2x2 slice, and every host has to run the recipe. So the JobSet launches
recipe/configs/qwen3_5_397b/perf.py on both and rank 0 attaches to it, the same
split the multi-host accuracy files use.

Text only: the checkpoint serves images, but a random-ids sweep sends none.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST
from sglang.test.tpu import model_config_main

register_tpu_ci(
    est_time=5400,
    suite="nightly-tpu-perf-multihost-qwen3-5-397b",
    nightly=True,
    labels=["multihost", "qwen3.5-397b", "perf"],
)

MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/configs/qwen3_5_397b/perf.py",
    "weights_pvc": "model-weights",
    "timeout_per_file": 5400,
}

BASE_URL = os.environ.get("BASE_URL", DEFAULT_URL_FOR_TEST)
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "3600"))

# 8192 in + 1024 out, the same shape the other models on this lane are measured
# at, so they sit on one axis of the Workload dropdown.
WORKLOAD = "8k1k"
# CONCUR must match CONCURRENCY in the launch config: the server is sized for
# it, and a client driving more would measure a pool-bound shape.
ISL, OSL, CONCUR = 8192, 1024, 16

MODEL = "Qwen/Qwen3.5-397B-A17B"
WEIGHTS = "/weights/Qwen3.5-397B-A17B"


MODEL_CONFIG = {
    "model": MODEL,
    "test_name": "qwen3-5-397b",
    # The dashboard's series join key: never renamed, never shown. A forced
    # rename is recorded in .github/tpu-ci/dashboard_aliases.json instead.
    "id": "perf-qwen3-5-397b",
    "display_name": "Qwen3.5 397B-A17B",
    "variant": "2host-TP16+EP16",
    "launch": {
        # The JobSet owns the server: it runs the recipe on every host of the
        # slice, which a per-test launcher on rank 0 cannot do.
        "external": True,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "perf-qwen3-5-397b",
    },
    "skip_unless": (
        os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1",
        "multi-host live-server not provided (the JobSet launches the server)",
    ),
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
                # random-ids + exact input_ids, matching the InferenceX harness
                # the published TPU numbers come from; a tokenizer round-trip
                # would otherwise move the real ISL.
                "DATASET_NAME": "random-ids",
                "TOKENIZE_PROMPT": "1",
                # Tokenizer from the local checkpoint (no HF fetch in the pod);
                # requests still carry the name the server serves under.
                "MODEL": WEIGHTS,
                "SERVED_MODEL_NAME": MODEL,
            },
        }
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
