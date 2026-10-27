"""WeightChecker checksum and compare on a non-CUDA device (CPU stands in for TPU)."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.utils.weight_checker import ChecksumInfo, WeightChecker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestChecksumWithoutCuda(CustomTestCase):
    def test_checksum_does_not_require_cuda(self):
        """The checksum action must work on hosts without CUDA, e.g. TPU engines
        checked by an RL trainer after every weight update."""
        model = nn.Linear(4, 4)
        enter_scope(self, published_topology())
        checker = WeightChecker(get_model=lambda: model)
        with patch.object(
            torch.cuda, "synchronize", side_effect=RuntimeError("no CUDA device")
        ):
            out = checker.handle("checksum", role="target")
        info = ChecksumInfo.model_validate(out)
        self.assertEqual(set(info.checksums), {"weight", "bias"})


class TestCompareWithoutCuda(CustomTestCase):
    """An RL trainer snapshots the engine weights, overwrites them and has the first weight
    update restore them; the compare must run where the weights live, e.g. on a TPU engine."""

    def setUp(self):
        enter_scope(self, published_topology())
        # engine weights are not trained, so the in-place reset needs no grad
        self.model = nn.Linear(4, 4).requires_grad_(False)
        self.checker = WeightChecker(get_model=lambda: self.model)
        no_cuda = patch.object(
            torch.Tensor, "cuda", side_effect=RuntimeError("no CUDA device")
        )
        no_cuda.start()
        self.addCleanup(no_cuda.stop)

    def _handle(self, action):
        return self.checker.handle(action, role="target")

    def test_restored_weights_compare_equal(self):
        restored = {k: v.clone() for k, v in self.model.state_dict().items()}
        self._handle("snapshot")
        self._handle("reset_tensors")
        self.model.load_state_dict(restored)
        self._handle("compare")

    def test_weights_left_overwritten_fail_the_compare(self):
        self._handle("snapshot")
        self._handle("reset_tensors")
        with self.assertRaisesRegex(Exception, "check tensor equality failed"):
            self._handle("compare")


if __name__ == "__main__":
    unittest.main()
