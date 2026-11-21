# SPDX-License-Identifier: Apache-2.0
"""CPU unit test: on the V4.1 TPU pool, ForwardBatch's no-op pad is skipped."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41 import memory_pool
from sglang_tpu.models.deepseek_v41.hooks import skip_tpu_noop_pad


class TestSkipNoopPad(CustomTestCase):
    def _pad(self, tensor, size, *, dsv41=True, **kwargs):
        with mock.patch.object(memory_pool, "_DSV41_POOL_ACTIVE", dsv41):
            return skip_tpu_noop_pad(
                ForwardBatch._pad_tensor_to_size,
                SimpleNamespace(),
                tensor,
                size,
                **kwargs,
            )

    def test_tensor_at_size_is_returned_as_is_on_the_v41_pool(self):
        t = torch.arange(4)
        self.assertIs(self._pad(t, 4), t)

    def test_other_models_keep_the_original_copy(self):
        t = torch.arange(4)
        out = self._pad(t, 4, dsv41=False)
        self.assertIsNot(out, t)
        torch.testing.assert_close(out, t)

    def test_shorter_tensor_is_padded_like_the_original(self):
        t = torch.arange(3)
        torch.testing.assert_close(self._pad(t, 5), torch.tensor([0, 1, 2, 0, 0]))
        torch.testing.assert_close(
            self._pad(t, 5, value=-1), torch.tensor([0, 1, 2, -1, -1])
        )


if __name__ == "__main__":
    unittest.main()
