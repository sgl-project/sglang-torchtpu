"""The ICI receiver must view every tensor back out of the trainer's packed bucket."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)

import unittest
from types import SimpleNamespace

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.rl.ici_receiver import receive_bucket, unpack_bucket
from sglang_tpu.rl.shared_slice import SharedSlice, SharedSliceSpec

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


class _FakeBackend:
    """Delivers the buffer queued for each source into the receiver's, recording (src, tag)."""

    def __init__(self, pending: dict[int, torch.Tensor]):
        self.pending, self.calls = pending, []

    def recv(self, buffers, src, tag):
        self.calls.append((src, tag))
        [buffer] = buffers
        buffer.copy_(self.pending.pop(src))
        return self

    def wait(self):
        pass


class TestReceiveBucket(CustomTestCase):
    def test_a_rank_receives_every_transfer_listed_for_it(self):
        """Trainer ranks may hold different tensors: rank 1 takes the dense bucket from one source and experts from another."""
        dense = [("model.norm.weight", torch.arange(6, dtype=torch.float32))]
        experts = [
            ("model.layers.0.mlp.experts.2.w1", torch.ones(3, 5, dtype=torch.bfloat16)),
            (
                "model.layers.0.mlp.experts.3.w1",
                torch.zeros(3, 5, dtype=torch.bfloat16),
            ),
        ]
        (dense_buffer, dense_entries), (expert_buffer, expert_entries) = (
            _pack_like_the_trainer(dense),
            _pack_like_the_trainer(experts),
        )
        backend = _FakeBackend({0: dense_buffer, 2: expert_buffer})
        shared_slice = SharedSlice(
            spec=SharedSliceSpec(rank_offset=4, size=8, store_host="h", store_port=1),
            slice_rank=5,  # job rank 1
            backend=backend,
        )
        context = SimpleNamespace(runners=[("target", SimpleNamespace(device="cpu"))])
        transfer = lambda src, buffer, entries: {
            "src_slice_rank": src,
            "num_bytes": buffer.numel() * buffer.element_size(),
            "tensors": entries,
        }
        payload = {
            "tag": 7,
            "receives": [
                [transfer(1, dense_buffer, dense_entries)],
                [
                    transfer(0, dense_buffer, dense_entries),
                    transfer(2, expert_buffer, expert_entries),
                ],
            ],
        }

        received = receive_bucket(
            context=context, payload=payload, shared_slice=shared_slice
        )

        self.assertEqual(backend.calls, [(0, 7), (2, 7)])
        self.assertEqual([n for n, _ in received], [n for n, _ in dense + experts])
        for (_, expected), (_, actual) in zip(dense + experts, received):
            self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
