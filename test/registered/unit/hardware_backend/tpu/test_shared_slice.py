"""CPU contracts of the ICI weight-update receiver (sglang_tpu.shared_slice)."""

import json
import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
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

    def test_enter_slice_env_on_a_further_host_keeps_the_host_device_index(self):
        """Slice ranks come in blocks of TORCH_TPU_DEVICES_PER_HOST per host, so the device
        torch_tpu pins to LOCAL_RANK is the slice rank modulo that; the slice rank itself
        carries the host's offset."""
        cases = [
            # (offset, size, job rank, devices per host) -> (slice rank, local rank)
            (
                (8, 16, 3, "8"),
                ("11", "3"),
            ),  # one TP8 engine on host 1 of a 2-host slice
            ((24, 32, 5, "8"), ("29", "5")),  # engine 1 of two, on host 3
            ((120, 128, 7, "8"), ("127", "7")),  # engine 7's last rank, 16 hosts
            (
                (8, 16, 5, "4"),
                ("13", "1"),
            ),  # 4 devices per host: slice rank 13 is device 1 of host 3
        ]
        for (offset, size, job_rank, devices_per_host), (
            slice_rank,
            local_rank,
        ) in cases:
            env = {
                "RANK": str(job_rank),
                "WORLD_SIZE": "8",
                "LOCAL_RANK": str(job_rank),
                "LOCAL_WORLD_SIZE": "8",
                shared_slice.DEVICES_PER_HOST_ENV_VAR: devices_per_host,
            }
            with (
                self.subTest(offset=offset, job_rank=job_rank),
                patch.dict(os.environ, env),
            ):
                job = shared_slice.enter_slice_env(
                    shared_slice.SharedSlice(offset, size, "h", 1)
                )
                self.assertEqual(job, shared_slice.JobRank(rank=job_rank, size=8))
                self.assertEqual(
                    (
                        os.environ["RANK"],
                        os.environ["LOCAL_RANK"],
                        os.environ["WORLD_SIZE"],
                    ),
                    (slice_rank, local_rank, str(size)),
                )

    def test_join_pins_the_host_device_and_restores_the_job_env(self):
        """Engine 1 of two (offset 24 of 32), TP rank 5 on host 3: the runtime starts with
        LOCAL_RANK 5 (device 5 of its host), the job's devices come from the store in job
        order, and afterwards RANK / WORLD_SIZE / LOCAL_RANK are the job's again."""
        seen = {}
        store = _FakeStore()
        for rank in range(32):
            store.set(f"device:{rank}", str(1000 + rank))

        def global_device_id():
            seen["env"] = {
                k: os.environ[k] for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")
            }
            return 77

        tpu_distributed = SimpleNamespace(
            global_device_id=global_device_id,
            create_process_group=lambda *args, **kwargs: seen.setdefault(
                "group", object()
            ),
        )
        distributed = ModuleType("torch_tpu._internal.distributed")
        distributed.tpu_distributed = tpu_distributed
        modules = {
            "torch_tpu": ModuleType("torch_tpu"),
            "torch_tpu._internal": ModuleType("torch_tpu._internal"),
            "torch_tpu._internal.distributed": distributed,
        }
        env = {
            "RANK": "5",
            "WORLD_SIZE": "8",
            "LOCAL_RANK": "5",
            "LOCAL_WORLD_SIZE": "8",
        }
        shared = shared_slice.SharedSlice(24, 32, "store-host", 30108)
        with (
            patch.dict(os.environ, env),
            patch.dict(sys.modules, modules),
            patch.object(shared_slice.dist, "TCPStore", return_value=store),
            patch.object(shared_slice, "PrefixStore", lambda prefix, s: s),
            patch.object(shared_slice, "_state", shared_slice._State()),
        ):
            os.environ.pop(shared_slice.DEVICES_PER_HOST_ENV_VAR, None)
            job = shared_slice.enter_slice_env(shared)
            shared_slice.join(shared, job)
            self.assertEqual(
                seen["env"], {"RANK": "29", "LOCAL_RANK": "5", "WORLD_SIZE": "32"}
            )
            self.assertTrue(shared_slice.joined())
            self.assertEqual(
                (shared_slice._state.slice_rank, shared_slice._state.shared),
                (29, shared),
            )
            self.assertEqual(store.get("device:29"), b"77")
            self.assertEqual(
                os.environ["TORCH_TPU_JOB_DEVICE_IDS"],
                ",".join("77" if r == 5 else str(1024 + r) for r in range(8)),
            )
            self.assertEqual(
                (
                    os.environ["RANK"],
                    os.environ["LOCAL_RANK"],
                    os.environ["WORLD_SIZE"],
                ),
                ("5", "5", "8"),
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
        shared = shared_slice.SharedSlice(4, 8, "h", 1)
        with patch.object(
            shared_slice,
            "_state",
            shared_slice._State(
                backend=object(), store=_FakeStore(), slice_rank=6, shared=shared
            ),
        ):
            receiver = TpuSRTPlatform.create_weight_update_receiver(
                None, shared_slice.BACKEND_NAME, tp_rank=2
            )
        self.assertIsInstance(receiver, WeightUpdateReceiver)
        self.assertEqual((receiver._src_slice_rank, receiver._slice_rank), (2, 6))

    def test_receiver_pairs_with_the_trainer_rank_half_a_slice_below(self):
        """The trainer holds slice ranks 0 .. size / 2 - 1 and the engines the rest in order, so
        the worker at slice rank size / 2 + i receives from trainer rank i: with one engine that
        is its TP rank, with n engines of k devices TP rank i % k of engine i // k."""
        cases = [
            # (slice rank, size, TP rank) -> trainer slice rank
            ((6, 8, 2), 2),  # one TP4 engine on one host (as before)
            ((11, 16, 3), 3),  # one TP8 engine on its own host
            (
                (29, 32, 5),
                13,
            ),  # engine 1 of two: TP rank 5 <- trainer rank 13 (replica 1, shard 5)
            ((127, 128, 7), 63),  # engine 7 of eight
            ((21, 32, 5), 5),  # a TP16 engine over two hosts, rank 5
        ]
        for (slice_rank, size, tp_rank), src in cases:
            state = shared_slice._State(
                backend=object(),
                store=_FakeStore(),
                slice_rank=slice_rank,
                shared=shared_slice.SharedSlice(size // 2, size, "h", 1),
            )
            with (
                self.subTest(slice_rank=slice_rank),
                patch.object(shared_slice, "_state", state),
            ):
                receiver = shared_slice.receiver(tp_rank)
            self.assertEqual(
                (receiver._src_slice_rank, receiver._slice_rank), (src, slice_rank)
            )

    def test_every_trainer_rank_feeds_exactly_one_engine_rank(self):
        """16 engines of 1 device, 2 of 8 and 1 of 16 behind a 16-rank trainer: each layout pairs
        the trainer's ranks 0..15 with the engine ranks one to one."""
        for engines, tp_size in ((16, 1), (2, 8), (1, 16)):
            sources = []
            for engine, tp_rank in (
                (e, t) for e in range(engines) for t in range(tp_size)
            ):
                slice_rank = 16 + engine * tp_size + tp_rank
                state = shared_slice._State(
                    backend=object(),
                    store=_FakeStore(),
                    slice_rank=slice_rank,
                    shared=shared_slice.SharedSlice(slice_rank - tp_rank, 32, "h", 1),
                )
                with patch.object(shared_slice, "_state", state):
                    sources.append(shared_slice.receiver(tp_rank)._src_slice_rank)
            with self.subTest(engines=engines, tp_size=tp_size):
                self.assertEqual(sources, list(range(16)))

    def test_a_receiver_needs_the_joined_slice(self):
        with patch.object(shared_slice, "_state", shared_slice._State()):
            with self.assertRaises(AssertionError):
                shared_slice.receiver(0)


if __name__ == "__main__":
    unittest.main()
