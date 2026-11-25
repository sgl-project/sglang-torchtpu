"""Muse Spark 1P1D DFlash on the unified KV pool, one TP8 host per engine: full GSM8K score.

The engines are the P/D kv-offload test's without its bounded prefill pool and host
offload, so a drop against the backbone's 0.9765 points at P/D transfer or the
unified pool rather than at the eval.
"""

import os
import socket
import unittest
from typing import Any

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import MUSE_SPARK
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
        "accuracy",
        "unified-kv",
    ],
)

TEST_NAME = "muse-spark-pd-singlehost-dflash-unified-gsm8k"


def self_hostname() -> str:
    # External single-host roles drop TPU_WORKER_HOSTNAMES and set PD_SELF_HOST.
    if os.environ.get("PD_SELF_HOST"):
        return os.environ["PD_SELF_HOST"]
    hostnames = [h for h in os.environ.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    return hostnames[0] if hostnames else socket.gethostname()


class TestMuseSparkPDSinglehostDflashUnifiedGSM8K(PDSingleHostTestCase):
    """Require the backbone GSM8K score and accepted DFlash drafts through P/D."""

    config_module = "recipe/configs/muse_spark/pd_singlehost_dflash_unified_gsm8k.py"
    log_path = "/tmp/pd_muse_spark_unified_gsm8k_prefill.log"
    decode_log_path = "/tmp/pd_muse_spark_unified_gsm8k_decode.log"
    self_hostname = staticmethod(self_hostname)
    tag = "pd-singlehost-muse-spark-unified-gsm8k"

    @classmethod
    def build_model_config(cls) -> dict[str, Any]:
        return {
            "model": MUSE_SPARK["served_model_name"],
            "test_name": TEST_NAME,
            "gates": [
                {
                    "dataset": "gsm8k",
                    "label": "gsm8k_full",
                    # Backbone nightly (test_tpu_muse_spark_gsm8k.py) measures 0.9765.
                    "baseline": 0.9765,
                    "atol": 0.05,
                    "num_examples": 1319,
                    "num_threads": 32,
                    "max_tokens": 4096,
                    "temperature": 0.0,
                    "reasoning_effort": "low",
                    # The separate-pool P/D twin measured a 4.78 mean on 200 questions.
                    "spec_accept_length_threshold": 4.0,
                },
            ],
        }

    def test_gsm8k(self):
        self.run_evaluations()


if __name__ == "__main__":
    unittest.main()
