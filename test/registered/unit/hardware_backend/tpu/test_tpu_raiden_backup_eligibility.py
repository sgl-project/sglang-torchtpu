"""The SWA member of a hybrid pool never stages raiden retraction backups.

Its payload is row-filtered on load and a staged handle restores whole pages,
so it takes the row path; the full member stages. Runs on CPU with no TPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import os
import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.mem_cache.kv_pool import TpuFusedMHATokenToKVPool


class TestRetractionBackupEligibility(CustomTestCase):
    def test_swa_member_never_stages(self):
        pool = SWAKVPool(
            size=32,
            size_swa=32,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=128,
            swa_attention_layer_ids=[0, 2],
            full_attention_layer_ids=[1, 3],
            device="cpu",
            token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
        )
        self.assertEqual(pool.swa_kv_pool.allocation_label, "SWA")
        with mock.patch.dict(os.environ, {"SGLANG_RAIDEN_RETRACT_HOST_GB": "1"}):
            self.assertIsNone(pool.swa_kv_pool._raiden_backup())


if __name__ == "__main__":
    unittest.main()
