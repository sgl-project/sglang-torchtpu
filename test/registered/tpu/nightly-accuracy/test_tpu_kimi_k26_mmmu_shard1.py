"""MMMU-Pro shard 1/2: 865 distinct questions on one 2x2x2 slice."""

import unittest
from functools import partial

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import KIMI_K26
from sglang.test.tpu.evals.mmmu_full import run_shard
from sglang.test.tpu.utils.accuracy_base import (
    AccuracyTestCase,
    live_server_only,
)

register_tpu_ci(
    est_time=11400,
    suite="nightly-tpu-multihost-accuracy",
    nightly=True,
    labels=["multihost", "kimi-k2.6", "accuracy"],
)
MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/configs/kimi_k26/mmmu.py",
    "weights_pvc": "model-weights",
    # Which accuracy-test-tpu.yml lane runs this file.
    "schedule": "nightly",
    "timeout_per_file": 11400,
}
CONFIG_MODULE = "recipe/configs/kimi_k26/mmmu.py"


def build_model_config():
    """Assemble this file's gates against the model's identity."""
    return {
        "id": "accuracy-kimi-k26-mmmu-shard1",
        "display_name": "Kimi K2.6 MMMU-Pro shard 1/2",
        "dashboard": "accuracy",
        "dashboard_model": "kimi-k26-mmmu-shard1",
        "model": KIMI_K26["served_model_name"],
        "test_name": "Kimi-K2.6-MMMU-shard1",
        "variant": KIMI_K26["variant"],
        "gates": [
            {
                "dataset": "mmmu_pro",
                "label": "mmmu_pro_shard1",
                "baseline": 0.791,  # measured on TPU
                "atol": 0.1,
                "runner": partial(
                    run_shard,
                    shard=1,
                    shards=2,
                    model=KIMI_K26["served_model_name"],
                    **KIMI_K26["sampling"],
                ),
            }
        ],
    }


@live_server_only
class TestKimiMMMUShard1(AccuracyTestCase):
    """Score this file's gates on the server the JobSet started."""

    config_module = CONFIG_MODULE
    tag = "kimi-mmmu-shard1-e2e"

    @classmethod
    def build_model_config(cls):
        return build_model_config()

    def test_accuracy(self):
        self.run_evaluations()


class TestKimiMMMUShard1Configuration(unittest.TestCase):
    """Host-side configuration tests, runnable without TPU hardware."""

    def test_the_jobset_launches_this_files_recipe(self):
        self.assertEqual(MULTIHOST_LAUNCH["launch_script"], CONFIG_MODULE)


if __name__ == "__main__":
    unittest.main()
