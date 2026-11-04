"""CPU contracts of the ICI weight-update receiver (sglang_tpu.shared_slice)."""

import json
import os
import unittest
from unittest.mock import patch

import torch

from sglang.srt.platforms.interface import SRTPlatform, WeightUpdateReceiver
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import shared_slice
from sglang_tpu.tpu_platform import TpuSRTPlatform

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _pack(tensors):
    """The trainer's packing: int16 words, zero-padded regions."""
    regions = shared_slice.packed_regions(
        [t.numel() for t in tensors], [t.dtype for t in tensors]
    )
    pieces = []
    for tensor, (_, size) in zip(tensors, regions):
        words = tensor.reshape(-1).view(torch.int16)
        pieces += [words, words.new_zeros(size // 2 - words.numel())]
    return torch.cat(pieces)


class _FakeStore:
    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value.encode() if isinstance(value, str) else value

    def get(self, key):
        return self.values[key]

    def delete_key(self, key):
        del self.values[key]


class _FakeBackend:
    """Delivers queued packed buffers into the receiver's buffer, recording (src, tag)."""

    def __init__(self):
        self.pending, self.calls = [], []

    def recv(self, buffers, src, tag):
        self.calls.append((src, tag))
        [buffer] = buffers
        buffer.copy_(self.pending.pop(0))
        return self

    def wait(self):
        pass


class TestPackedRegions(CustomTestCase):
    def test_regions_are_whole_aligned_blocks(self):
        regions = shared_slice.packed_regions(
            [1, 2048, 2049, 0],
            [torch.float32, torch.bfloat16, torch.bfloat16, torch.float32],
        )
        # 4 B -> one block, 4096 B -> one block, 4098 B -> two blocks, empty -> still one block
        self.assertEqual(
            regions, [(0, 4096), (4096, 4096), (8192, 8192), (16384, 4096)]
        )


class TestSharedSliceReceiver(CustomTestCase):
    def test_receive_round_trips_tensors_and_names(self):
        backend, store = _FakeBackend(), _FakeStore()
        receiver = shared_slice.SharedSliceReceiver(
            backend, store, src_slice_rank=1, slice_rank=5
        )
        buckets = [
            [
                (
                    "model.layers.0.mlp.experts.4.down_proj.weight",
                    torch.randn(3, 5, dtype=torch.bfloat16),
                ),
                (
                    "model.layers.0.input_layernorm.weight",
                    torch.randn(7, dtype=torch.float32),
                ),
            ],
            [("lm_head.weight", torch.randn(2, 4096, dtype=torch.bfloat16))],
        ]
        for tag, bucket in enumerate(buckets, start=1):
            names, tensors = zip(*bucket)
            backend.pending.append(_pack(list(tensors)))
            store.set(shared_slice.bucket_names_key(1, 5, tag), json.dumps(list(names)))
            received = receiver.receive(
                ["names of trainer rank 0, ignored"] * len(bucket),
                [str(t.dtype).removeprefix("torch.") for t in tensors],
                [list(t.shape) for t in tensors],
                device="cpu",
            )
            self.assertEqual([name for name, _ in received], list(names))
            for (_, got), want in zip(received, tensors):
                self.assertEqual(got.dtype, want.dtype)
                self.assertTrue(
                    torch.equal(got.view(torch.int16), want.view(torch.int16))
                )
        # one tag per bucket, never reused, and each bucket's names are consumed
        self.assertEqual(backend.calls, [(1, 1), (1, 2)])
        self.assertEqual(store.values, {})

    def test_name_count_must_match_the_bucket(self):
        backend, store = _FakeBackend(), _FakeStore()
        receiver = shared_slice.SharedSliceReceiver(
            backend, store, src_slice_rank=0, slice_rank=4
        )
        tensor = torch.zeros(4, dtype=torch.float32)
        backend.pending.append(_pack([tensor]))
        store.set(shared_slice.bucket_names_key(0, 4, 1), json.dumps(["a", "b"]))
        with self.assertRaises(AssertionError):
            receiver.receive(["a"], ["float32"], [[4]], device="cpu")


class TestSliceEnvironment(CustomTestCase):
    def test_spec_parses_the_env_var(self):
        with patch.dict(os.environ, {shared_slice.ENV_VAR: "4,8,10.0.0.1:29600"}):
            self.assertEqual(
                shared_slice.spec(), shared_slice.SharedSlice(4, 8, "10.0.0.1", 29600)
            )
        with patch.dict(os.environ, {shared_slice.ENV_VAR: ""}):
            self.assertIsNone(shared_slice.spec())

    def test_enter_slice_env_swaps_in_the_slice_ranks(self):
        env = {
            "RANK": "2",
            "WORLD_SIZE": "4",
            "LOCAL_RANK": "2",
            "LOCAL_WORLD_SIZE": "4",
        }
        with patch.dict(os.environ, env):
            job = shared_slice.enter_slice_env(shared_slice.SharedSlice(4, 8, "h", 1))
            self.assertEqual(job, shared_slice.JobRank(rank=2, size=4))
            self.assertEqual(
                (
                    os.environ["RANK"],
                    os.environ["LOCAL_RANK"],
                    os.environ["WORLD_SIZE"],
                    os.environ["LOCAL_WORLD_SIZE"],
                ),
                ("6", "6", "8", "8"),
            )


class TestReceiverFactory(CustomTestCase):
    def test_other_backends_keep_torch_distributed_groups(self):
        self.assertIsNone(
            SRTPlatform.create_weight_update_receiver(None, "nccl", tp_rank=0)
        )
        self.assertIsNone(
            TpuSRTPlatform.create_weight_update_receiver(None, "gloo", tp_rank=0)
        )

    def test_shared_slice_backend_returns_the_receiver(self):
        with patch.object(
            shared_slice,
            "_state",
            shared_slice._State(backend=object(), store=_FakeStore(), slice_rank=6),
        ):
            receiver = TpuSRTPlatform.create_weight_update_receiver(
                None, shared_slice.BACKEND_NAME, tp_rank=2
            )
        self.assertIsInstance(receiver, WeightUpdateReceiver)
        self.assertEqual((receiver._src_slice_rank, receiver._slice_rank), (2, 6))


if __name__ == "__main__":
    unittest.main()
