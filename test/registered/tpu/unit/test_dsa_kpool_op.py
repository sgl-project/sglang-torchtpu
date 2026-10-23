# SPDX-License-Identifier: Apache-2.0
"""Exercise the serving custom op, including cache mutation under torch.compile."""

import os
import unittest

# TorchTPU owns the device; JAX exports the custom call on the CPU.
os.environ["JAX_PLATFORMS"] = "cpu"

import torch
import torch_tpu  # noqa: F401

from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.dsa_indexer_tpu import build_dsa_kpool_topk_op

register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit", chips=1)


class TestKpoolServingOp(unittest.TestCase):
    def test_eager_and_compiled_update_the_original_caches(self):
        op = build_dsa_kpool_topk_op()
        pages = torch.tensor([1, 3, 5, 7, 2, 4, 6, 8], dtype=torch.int32)
        ape = torch.zeros(4, 128, dtype=torch.float32, device="tpu")

        def run(*args):
            return op(*args)

        compiled = torch.compile(run, backend="tpu", fullgraph=True, dynamic=False)
        results = []
        for call in (run, compiled):
            raw_cpu = torch.zeros(9, 128, 256, dtype=torch.bfloat16)
            pooled_cpu = torch.full((9, 32, 4, 256), 173, dtype=torch.uint8)
            raw, pooled = raw_cpu.to("tpu"), pooled_cpu.to("tpu")
            lens = [0, 0]
            outputs = []
            # Ragged prefill, uniform multi-token verify, then decode. Verify
            # closes pools containing raw rows from both this and the last call.
            for step, (new, decode_width, bound) in enumerate(
                (((15, 7), 1, 32), ((6, 6), 6, 6), ((1, 1), 1, 1))
            ):
                tokens = 32
                key = torch.zeros(tokens, 128, dtype=torch.bfloat16)
                key[: new[0]] = step + 1
                key[new[0] : sum(new)] = -(step + 1) / 2
                cu = torch.tensor([0, new[0], sum(new)], dtype=torch.int32)
                after = [lens[r] + new[r] for r in range(2)]
                distribution = [0, 0, 2] if step == 0 else [2, 2, 2]
                # An independent reference for constant channels and zero gate:
                # average four raw rows, round to BF16, rotate, and quantize.
                for r in range(2):
                    for i in range(new[r]):
                        pos = lens[r] + i
                        page = int(pages[r * 4 + pos // 128])
                        raw_cpu[page, pos % 128, :128] = key[int(cu[r]) + i]
                    for end in range((lens[r] // 4) * 4 + 3, after[r], 4):
                        page = int(pages[r * 4 + end // 128])
                        value = raw_cpu[page, end % 128 - 3 : end % 128 + 1, 0]
                        mean = value.float().mean().bfloat16().float()
                        rotated = mean * (128**0.5)
                        exponent = torch.ceil(
                            torch.log2(rotated.abs().clamp_min(1e-4) / 448)
                        )
                        fp8 = (rotated / torch.exp2(exponent)).to(torch.float8_e4m3fn)
                        record = torch.zeros(256, dtype=torch.uint8)
                        record[0] = fp8.view(torch.uint8)
                        record[128] = (exponent + 127).to(torch.uint8)
                        pool_id = end // 4
                        p_page = int(pages[r * 4 + pool_id // 128])
                        pooled_cpu[p_page, pool_id % 128 // 4, pool_id % 4] = record
                query = torch.ones(tokens, 8, 128, dtype=torch.bfloat16, device="tpu")
                out = call(
                    query,
                    key.to("tpu"),
                    torch.zeros_like(key, device="tpu"),
                    torch.ones(tokens, 8, dtype=torch.float32, device="tpu"),
                    ape,
                    raw,
                    pooled,
                    torch.tensor(after, dtype=torch.int32, device="tpu"),
                    pages.to("tpu"),
                    cu.to("tpu"),
                    torch.tensor(distribution, dtype=torch.int32, device="tpu"),
                    1.0,
                    64,
                    4,
                    128,
                    decode_width,
                    bound,
                )
                torch.accelerator.synchronize()
                torch.testing.assert_close(raw.cpu(), raw_cpu, rtol=0, atol=0)
                torch.testing.assert_close(pooled.cpu(), pooled_cpu, rtol=0, atol=0)
                out = out.cpu()
                self.assertEqual(tuple(out.shape), (tokens, 128))
                self.assertTrue(torch.all(out[sum(new) :] == -1))
                for r in range(2):
                    for i in range(new[r]):
                        row = out[int(cu[r]) + i]
                        valid = row[row >= 0]
                        torch.testing.assert_close(
                            valid.sort().values,
                            torch.arange(lens[r] + i + 1, dtype=torch.int32),
                            rtol=0,
                            atol=0,
                        )
                outputs.append(out)
                lens = after
            results.append(outputs)
        for eager, compiled_out in zip(*results):
            torch.testing.assert_close(eager, compiled_out, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
