"""Muse Spark 1P1D DFlash, one TP8 host per engine, prefix cache and FP8 KV offload.

The prefill pod also runs the mini-LB and the gates through it.
"""

import os
import socket
import unittest
from typing import Any

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import MUSE_SPARK
from sglang.test.tpu.utils.kv_offload_probe import kv_offload_runner
from sglang.test.tpu.utils.pd_singlehost_base import PDSingleHostTestCase

register_tpu_ci(
    est_time=7200,
    suite="tpu-integration",
    nightly=True,
    labels=[
        "singlehost",
        "muse-spark-open-816b-a42b",
        "pd",
        "spec",
        "dflash",
        "prefix-cache",
        "kv-offload",
    ],
)

TEST_NAME = "muse-spark-pd-singlehost-dflash"


def self_hostname() -> str:
    # External single-host roles drop TPU_WORKER_HOSTNAMES and set PD_SELF_HOST.
    if os.environ.get("PD_SELF_HOST"):
        return os.environ["PD_SELF_HOST"]
    hostnames = [h for h in os.environ.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    return hostnames[0] if hostnames else socket.gethostname()


class TestMuseSparkPDSinglehostDflash(PDSingleHostTestCase):
    """Require accepted DFlash drafts and a complete prefill host-cache reload."""

    config_module = (
        "recipe/configs/muse_spark/pd_singlehost_prefix_cache_kv_offload_dflash.py"
    )
    log_path = "/tmp/pd_muse_spark_prefill.log"
    decode_log_path = "/tmp/pd_muse_spark_decode.log"
    self_hostname = staticmethod(self_hostname)
    tag = "pd-singlehost-muse-spark"

    @classmethod
    def build_model_config(cls) -> dict[str, Any]:
        return {
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
                    # Measured 4.78 mean over 200 (per-request 3.71..6.08).
                    "spec_accept_length_threshold": 4.0,
                },
            ],
        }

    def test_prefix_cache_kv_offload_dflash(self):
        self.run_evaluations()


if __name__ == "__main__":
    unittest.main()
