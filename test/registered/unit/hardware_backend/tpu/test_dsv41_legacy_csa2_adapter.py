# SPDX-License-Identifier: Apache-2.0
"""Regression for the preserved modeling snapshot's removed SWA kernel APIs.

Import must work without vLLM/TorchTPU, and attempted legacy SWA use must fail
before device-op registration, cache mutation, or an empty-batch success path.
This does not validate TPU execution or enable the disconnected snapshot.
"""

import inspect
import unittest

import torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from sglang_tpu.models.deepseek_v41.modeling import csa2_adapter

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLegacyCSA2Adapter(CustomTestCase):
    def test_legacy_swa_fails_before_cache_or_device_access(self):
        # None cache/metadata deliberately cannot satisfy a hidden fallback.
        # An empty query must not bypass the unsupported-contract diagnostic.
        functions = (
            csa2_adapter.initialize_csa2_ops,
            csa2_adapter.publish_swa,
            csa2_adapter.prepare_swa,
            csa2_adapter.prepare_publish_swa,
            csa2_adapter.attention,
            csa2_adapter._publish_swa,
            csa2_adapter._publish_swa_banked,
            csa2_adapter._prepare_publish_swa,
            csa2_adapter._prepare_publish_swa_banked,
        )
        for function in functions:
            with self.subTest(function=function.__name__):
                args = [None] * len(inspect.signature(function).parameters)
                if function is csa2_adapter.attention:
                    args[0] = torch.empty((0, 8, 512), dtype=torch.bfloat16)
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"private BF16 SWA pages.*flat INT32.*allocation, binding.*"
                    r"not connected to the active SGLang",
                ):
                    function(*args)
        self.assertEqual(csa2_adapter._ops, {})


if __name__ == "__main__":
    unittest.main()
