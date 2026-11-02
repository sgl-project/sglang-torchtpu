"""Muse Spark single-host DFlash, prefix cache and FP8 KV offload."""

import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import MUSE_SPARK, run_model_config
from sglang.test.tpu.utils.kv_offload_probe import kv_offload_runner
from sglang.test.tpu.utils.launcher_test_case import TPULauncherTestCase

register_tpu_ci(
    # The external suite's 1 h per file covers it: 2,878 s in-project.
    est_time=3600,
    suite="external-nightly-1host",
    nightly=True,
    labels=[
        "singlehost",
        "muse-spark-open-816b-a42b",
        "spec",
        "dflash",
        "prefix-cache",
        "kv-offload",
    ],
)

TEST_NAME = "muse-spark-singlehost-prefix-cache-kv-offload-dflash"


class TestMuseSparkSinglehostDflash(TPULauncherTestCase):
    """Require accepted DFlash drafts and a complete host-cache reload.

    Muse runs no DP attention: its NVFP4 fused-RS MoE requires it off.
    """

    config_module = (
        "recipe/configs/muse_spark/singlehost_prefix_cache_kv_offload_dflash.py"
    )
    tag = "muse-spark-singlehost-dflash"
    health_timeout = 5400
    base_url = "http://127.0.0.1:30000"

    def test_prefix_cache_kv_offload_dflash(self):
        run_model_config(
            {
                "model": MUSE_SPARK["served_model_name"],
                "test_name": TEST_NAME,
                "gates": [
                    {
                        "dataset": "kv_offload",
                        "label": "kv_offload_dflash",
                        "runner": kv_offload_runner(
                            test_name=TEST_NAME,
                            device_pool_tokens=8192,
                            page_size=128,
                            probe_tokens=2048,
                            hicache_ratio=3,
                            spec_decoding=True,
                        ),
                    },
                    {
                        "dataset": "gsm8k",
                        "baseline": 0.98,
                        "atol": 0.10,
                        "num_examples": 200,
                        "num_threads": 8,
                        "max_tokens": 4096,
                        "temperature": 0.0,
                        "reasoning_effort": "low",
                        # Measured 4.80 mean over 200 (per-request 3.64..5.82).
                        "spec_accept_length_threshold": 4.0,
                    },
                ],
            },
            base_url=self.base_url,
        )


if __name__ == "__main__":
    unittest.main()
