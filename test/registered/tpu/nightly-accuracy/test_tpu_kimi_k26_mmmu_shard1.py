"""MMMU-Pro shard 1/2: 865 distinct questions on one 2x2x2 slice."""

import os
from functools import partial

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.mmmu_full import run_shard

register_tpu_ci(
    est_time=14400,
    suite="nightly-tpu-multihost-accuracy",
    nightly=True,
    labels=["multihost", "kimi-k2.6", "accuracy"],
)
MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/kimi_k26_mmmu.sh",
    "weights_pvc": "model-weights-rox",
    "timeout_per_file": 14400,
}
MODEL_CONFIG = {
    "id": "accuracy-kimi-k26-mmmu-shard1",
    "display_name": "Kimi K2.6 MMMU-Pro shard 1/2",
    "dashboard": "accuracy",
    "dashboard_model": "kimi-k26-mmmu-shard1",
    "model": "/weights/Kimi-K2.6",
    "test_name": "Kimi-K2.6-MMMU-shard1",
    "variant": "TP16-EP16-DP1",
    "launch": {"external": True, "health_timeout": 3600},
    "skip_unless": (
        os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1",
        "live server required",
    ),
    "gates": [
        {
            "dataset": "mmmu_pro",
            "label": "mmmu_pro_shard1",
            "report_only": True,
            "runner": partial(run_shard, shard=1, shards=2),
        }
    ],
}
if __name__ == "__main__":
    model_config_main(MODEL_CONFIG)
