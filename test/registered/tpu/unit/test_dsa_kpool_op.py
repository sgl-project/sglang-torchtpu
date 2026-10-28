# SPDX-License-Identifier: Apache-2.0
"""The serving custom op must update the caches it is given, eager and compiled.

Pooling numerics are checked in test_dsa_kpool_topk.py; this guards the torch
side of the op: the raw and pooled caches are donated and aliased, so a lost
alias would leave the caller's tensors stale.
"""

import os
import unittest

# TorchTPU owns the device; JAX exports the custom call on the CPU.
os.environ["JAX_PLATFORMS"] = "cpu"

import torch
import torch_tpu  # noqa: F401

from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.dsa_indexer_tpu import build_dsa_kpool_topk_op

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

_FILL = 173
_TOKENS = 32


class TestKpoolServingOp(unittest.TestCase):
    def test_eager_and_compiled_update_the_original_caches(self):
        op = build_dsa_kpool_topk_op()
        pages = torch.tensor([1, 3, 5, 7, 2, 4, 6, 8], dtype=torch.int32)

        def run(*args):
            return op(*args)

        compiled = torch.compile(run, backend="tpu", fullgraph=True, dynamic=False)
        results = []
        for call in (run, compiled):
            raw = torch.zeros(9, 128, 256, dtype=torch.bfloat16, device="tpu")
            pooled = torch.full((9, 32, 4, 256), _FILL, dtype=torch.uint8, device="tpu")
            lens = [0, 0]
            outputs = []
            # Ragged prefill (token-bucket bound), uniform verify (verify-width
            # bound), then decode, with the bounds serving passes.
            for step, (new, decode_width, bound) in enumerate(
                (((15, 7), 1, _TOKENS), ((6, 6), 6, 6), ((1, 1), 1, 1))
            ):
                after = [lens[r] + new[r] for r in range(2)]
                key = torch.zeros(_TOKENS, 128, dtype=torch.bfloat16)
                key[: sum(new)] = step + 1
                out = call(
                    torch.ones(_TOKENS, 8, 128, dtype=torch.bfloat16, device="tpu"),
                    key.to("tpu"),
                    torch.zeros(_TOKENS, 128, dtype=torch.bfloat16, device="tpu"),
                    torch.ones(_TOKENS, 8, dtype=torch.float32, device="tpu"),
                    torch.zeros(4, 128, dtype=torch.float32, device="tpu"),
                    raw,
                    pooled,
                    torch.tensor(after, dtype=torch.int32, device="tpu"),
                    pages.to("tpu"),
                    torch.tensor([0, new[0], sum(new)], dtype=torch.int32).to("tpu"),
                    torch.tensor(
                        [0, 0, 2] if step == 0 else [2, 2, 2], dtype=torch.int32
                    ).to("tpu"),
                    1.0,
                    64,
                    4,
                    128,
                    decode_width,
                    bound,
                )
                torch.accelerator.synchronize()
                raw_cpu, pooled_cpu = raw.cpu(), pooled.cpu()
                for r in range(2):
                    for pos in range(lens[r], after[r]):
                        page = int(pages[r * 4 + pos // 128])
                        # The caller's raw tensor holds this step's key rows.
                        self.assertEqual(float(raw_cpu[page, pos % 128, 0]), step + 1)
                        if pos % 4 == 3:
                            # The caller's pooled tensor holds the closed pool.
                            pool = pos // 4
                            p_page = int(pages[r * 4 + pool // 128])
                            record = pooled_cpu[p_page, pool % 128 // 4, pool % 4]
                            self.assertFalse(torch.all(record == _FILL))
                outputs.append((out.cpu(), raw_cpu, pooled_cpu))
                lens = after
            results.append(outputs)
        for eager, compiled_out in zip(*results):
            for a, b in zip(eager, compiled_out):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
