"""Qwen3-VL-30B-A3B single-host MMMU-Pro (vision) gate.

The other TPU multimodal nightlies serve the dense Qwen3-VL-8B, so nothing
covered a ViT prefill feeding a fused-MoE decoder. This serves the MoE
checkpoint (tp8 + ep8) and scores an image benchmark: a vision embedding,
deepstack or expert-routing regression shows up as a score drop.
"""

import os

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

register_tpu_ci(
    # A cold tpu7x 2x2x1 run measured ~35 min: ~21 min to ready, ~10 min of
    # sgl-eval dataset prep (HF_HOME is per-pod), ~5 min of eval.
    est_time=3000,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "qwen3-vl-30b-a3b", "multimodal", "compile-tracer"],
)

SERVED_MODEL_NAME = "qwen3-vl-30b-a3b"
MODEL = "/weights/Qwen3-VL-30B-A3B-Instruct"
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
# Budget: the ViT and VL decoder warmups compile concurrently under the
# recipe's direct tracer, ~8 min cold (~39 min under dynamo); the 45 min
# default also covers a dynamo comparison run.
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "2700"))

# Deployment environment only: the serving settings are the recipe's, so this
# gate and the P/D gate serve the same configuration.
LAUNCH_ENV = {
    "MODEL": MODEL,
    # A cold start writes ~49 GB of per-rank executables to the tier-2 compile
    # cache and the same again to its tier-3 backup. CI keeps both in the pod's
    # 64Gi /dev/shm, and a fresh pod has nothing to reuse, so skip the cache.
    "ENABLE_COMPILE_CACHE": "0",
    # common_env.sh defaults this to +recompiles, and dynamo reprints the whole
    # cached guard list per recompile. Serving-time recompiles stay fatal via
    # the scheduler's fail_on_recompile stance.
    "TORCH_LOGS": "-dynamo",
    "SERVED_MODEL_NAME": SERVED_MODEL_NAME,
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "qwen3-vl-30b-a3b-mmmu-pro-vision",
    "variant": "TP8+EP8",
    "launch": {
        "recipe": "recipe/qwen3_vl_30b_a3b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "qwen3-vl-30b-a3b-mmmu-e2e",
    },
    "gates": [
        {
            "dataset": "mmmu_pro_vision",
            "label": "mmmu_pro_vision",
            # Two tpu7x 2x2x1 runs scored 0.445 and 0.455, with the same answer
            # on 198/200 items. The floor leaves room for numerics changes that
            # flip a few answers; a broken vision or expert path should land
            # well below it.
            "baseline": 0.45,
            "atol": 0.08,
            "num_examples": 200,
            "num_threads": 8,
            # Greedy CoT answers are long: 512 truncated half of them (each
            # scored 0), 4096 truncates 29%, mostly answers stuck re-checking.
            "max_tokens": 4096,
            "temperature": 0.0,
        }
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
