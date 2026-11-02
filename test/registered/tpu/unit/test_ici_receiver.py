"""The ICI receiver must view every tensor back out of the trainer's packed bucket."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)

import unittest

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.weight_sync.ici_receiver import unpack_bucket

# miles protocols/ici.py: each tensor starts on a 4 KiB boundary of one int16 buffer.
_ALIGN_BYTES = 4096


def _pack_like_the_trainer(
    named_tensors: list[tuple[str, torch.Tensor]],
) -> tuple[torch.Tensor, list[list]]:
    raw, entries, offset = [], [], 0
    for name, tensor in named_tensors:
        data = tensor.contiguous().reshape(-1).view(torch.uint8)
        region = max(_ALIGN_BYTES, -(-data.numel() // _ALIGN_BYTES) * _ALIGN_BYTES)
        padding = torch.zeros(region - data.numel(), dtype=torch.uint8)
        raw.append(torch.cat([data, padding]))
        dtype_name = str(tensor.dtype).removeprefix("torch.")
        entries.append([name, dtype_name, list(tensor.shape), offset])
        offset += region
    return torch.cat(raw).view(torch.int16), entries


class TestUnpackBucket(CustomTestCase):
    def test_every_entry_views_back_its_exact_tensor(self):
        """Odd byte lengths, wide dtypes, a transposed source and an empty tensor."""
        gen = torch.Generator().manual_seed(0)
        named_tensors = [
            ("odd_bf16", torch.randn(3, 5, generator=gen).to(torch.bfloat16)),
            ("transposed_fp32", torch.randn(4, 6, generator=gen).t()),
            ("odd_uint8", torch.randint(255, (7,), generator=gen, dtype=torch.uint8)),
            ("empty", torch.empty(0, 4, dtype=torch.bfloat16)),
            ("past_one_region_int64", torch.arange(600, dtype=torch.int64)),
        ]
        buffer, entries = _pack_like_the_trainer(named_tensors)

        unpacked = unpack_bucket(buffer, entries)

        self.assertEqual([n for n, _ in unpacked], [n for n, _ in named_tensors])
        for (name, expected), (_, actual) in zip(named_tensors, unpacked):
            with self.subTest(name=name):
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
