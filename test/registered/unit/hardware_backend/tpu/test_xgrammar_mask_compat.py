"""TPU XGrammar masking preserves signed bits, selected rows and padded tails."""

import importlib
import unittest

import torch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.patches import (
    _register_tpu_compile_backend,
    _unwrap_xgrammar_compile_fns,
)

register_tpu_ci(
    est_time=20,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "infra", "agentic", "compile"],
)


class TestXGrammarMaskCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The sampler factory applies this patch after distributed initialization.
        _register_tpu_compile_backend()
        _unwrap_xgrammar_compile_fns()
        cls.module = importlib.import_module(
            "xgrammar.kernels.apply_token_bitmask_inplace_torch_compile"
        )

    def _check_masks(self, device):
        for vocab in (31, 32, 33, 65, 202048):
            width = (vocab + 31) // 32
            allowed = torch.arange(width * 32).expand(3, -1) % 3 == 1
            allowed[0] = True
            allowed[1] = False
            allowed[2, 31] = True
            shifts = 2 ** torch.arange(32, dtype=torch.int64)
            packed = (
                (allowed.reshape(3, width, 32).long() * shifts).sum(-1).to(torch.int32)
            )
            source = torch.arange(3 * (vocab + 7), dtype=torch.float32).reshape(
                3, vocab + 7
            )
            for indices in (None, [2, 0]):
                with self.subTest(device=device, vocab=vocab, indices=indices):
                    expected = source.clone()
                    for row in range(3) if indices is None else indices:
                        expected[row, :vocab].masked_fill_(
                            ~allowed[row, :vocab], -float("inf")
                        )
                    actual = source.to(device)
                    self.module.apply_token_bitmask_inplace_torch_compile(
                        actual, packed.to(device), vocab, indices
                    )
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def test_cpu_reference_semantics(self):
        self._check_masks("cpu")

    @unittest.skipUnless(hasattr(torch, "tpu"), "requires TorchTPU")
    def test_tpu_lowering(self):
        self._check_masks("tpu")


if __name__ == "__main__":
    unittest.main()
