# SPDX-License-Identifier: Apache-2.0
"""Exercise the opt-in serving builders, CPU export, and donated Torch cache.

Run after raw-JAX tests in a fresh process: TorchTPU owns PJRT here, while
JAX exports against CPU. Both ordinary and token-split ops must honor the
same frozen selector and mutate the caller's original uint8 cache.
"""

import os
import unittest

os.environ["JAX_PLATFORMS"] = "cpu"

import torch
import torch_tpu  # noqa: F401
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

from sglang_tpu.dsa_indexer_tpu import (
    build_dsa_index_topk_op,
    build_dsa_index_topk_split_op,
)
from sglang_tpu.envs import envs as tpu_envs

register_tpu_ci(
    est_time=90,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "glm-5.2", "compile", "spec"],
)


class TestDsaRadixServingOp(CustomTestCase):
    def test_eager_compiled_and_split_preserve_cache_alias_and_padding(self):
        with tpu_envs.SGLANG_TPU_USE_DSA_RADIX_TOPK.override(True):
            full = build_dsa_index_topk_op()
            split = build_dsa_index_topk_split_op()

        def run(*args):
            return full(*args)

        compiled = torch.compile(run, backend="tpu", fullgraph=True, dynamic=False)
        results = []
        for call in (run, compiled):
            query = torch.ones((4, 8, 128), dtype=torch.bfloat16, device="tpu")
            key = torch.ones((4, 128), dtype=torch.bfloat16, device="tpu")
            weights = torch.ones((4, 8), dtype=torch.float32, device="tpu")
            lens = torch.ones((4,), dtype=torch.int32, device="tpu")
            pages = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="tpu")
            cu = torch.arange(5, dtype=torch.int32, device="tpu")
            distribution = torch.tensor([4, 4, 4], dtype=torch.int32, device="tpu")
            cache = torch.full((5, 16, 4, 256), 173, dtype=torch.uint8, device="tpu")
            args = (query, key, weights, cache, lens, pages, cu, distribution)
            out = call(*args, 1.0, 2048, 1, "decode", "xla")
            torch.accelerator.synchronize()
            cpu = out.cpu()
            self.assertEqual(cpu.dtype, torch.int32)
            torch.testing.assert_close(cpu[:, 0], torch.zeros(4, dtype=torch.int32))
            torch.testing.assert_close(
                cpu[:, 1:], torch.full((4, 2047), -1, dtype=torch.int32)
            )
            updated = cache.cpu()
            for page in (1, 2, 3, 4):
                self.assertFalse(torch.all(updated[page, 0, 0] == 173))
            # Token splitting must use the same frozen candidate and cache ABI.
            sliced = split(
                *args,
                torch.zeros(1, dtype=torch.int32, device="tpu"),
                1.0,
                2048,
                1,
                "decode",
                "xla",
                4,
            )
            torch.accelerator.synchronize()
            torch.testing.assert_close(sliced.cpu(), cpu, rtol=0, atol=0)
            results.append((cpu, updated))
        for eager, traced in zip(results[0], results[1]):
            torch.testing.assert_close(eager, traced, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
