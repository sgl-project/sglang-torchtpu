"""Native radix checkpoints, including an intermediate prefill boundary."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=180, suite="stage-a-test-tpu-unit")


class TestGDNPrefixCache(CustomTestCase):
    def setUp(self):
        from sglang_tpu.attention_gdn_pallas import (
            build_gdn_pallas_op,
        )

        self.op = build_gdn_pallas_op()
        torch.manual_seed(7)
        self.nk, self.nv, self.dk, self.dv, self.k = 2, 4, 128, 128, 4
        self.dim = self.nk * self.dk * 2 + self.nv * self.dv
        self.weights = [
            torch.randn(self.dim, 1, self.k).to("tpu") * 0.1,
            None,
            torch.randn(self.nv).to("tpu"),
            torch.randn(self.nv).to("tpu"),
        ]
        self.inputs = [
            torch.randn(193, self.dim).to("tpu"),
            torch.randn(193, self.nv).to("tpu"),
            torch.randn(193, self.nv).to("tpu"),
        ]

    def states(self):
        return (
            torch.zeros(7, self.k - 1, self.dim, device="tpu"),
            torch.zeros(7, self.nv, self.dk, self.dv, device="tpu"),
        )

    def run_op(self, states, start, end, slot, *, prefix=0, track=None, decode=False):
        def ids(x):
            return torch.tensor(x, dtype=torch.int32, device="tpu")

        return self.op(
            *(t[start:end] for t in self.inputs),
            *states,
            *self.weights,
            ids([0, end - start]),
            ids([slot]),
            ids([1 if decode else 0, 1 if decode else 0, 1]),
            ids([prefix + end - start]),
            self.nk,
            self.nv,
            self.dk,
            self.dv,
            self.k,
            track_indices=ids([track[0]]) if track else None,
            track_end=ids([track[1]]) if track else None,
        )

    def test_prefill_snapshot_and_branch(self):
        # The saved state must be after 128 tokens, not after the complete
        # 193-token prefill. Resume two branches from that immutable snapshot.
        full, reference = self.states(), self.states()
        output = self.run_op(full, 0, 193, 1, track=(2, 128))
        self.run_op(reference, 0, 128, 1)
        for actual, expected in zip(full, reference):
            torch.testing.assert_close(
                actual[2].cpu(), expected[1].cpu(), atol=2e-5, rtol=2e-5
            )
        from sglang_tpu.mamba_pool import (
            TpuMambaPool,
            TpuPerLayerState,
        )

        pool = TpuMambaPool.__new__(TpuMambaPool)
        pool.device = "tpu"
        pool.checkpoint_width = 1
        pool.mamba_cache = TpuPerLayerState(conv=[[full[0]]], temporal=[full[1]])
        snapshot = [t[2].cpu().clone() for t in full]
        pool.copy_from(torch.tensor([2, 2]), torch.tensor([3, 4]))
        resumed = self.run_op(full, 128, 193, 3, prefix=128)
        torch.testing.assert_close(
            resumed.cpu(), output[128:].cpu(), atol=2e-5, rtol=2e-5
        )
        self.run_op(full, 128, 129, 4, prefix=128, decode=True)
        for state, saved in zip(full, snapshot):
            torch.testing.assert_close(state[2].cpu(), saved, atol=0, rtol=0)
            torch.testing.assert_close(
                state[3].cpu(), state[1].cpu(), atol=2e-5, rtol=2e-5
            )

    def test_copy_buckets_preserve_states_without_recompilation(self):
        from torch._dynamo.utils import counters

        from sglang_tpu.mamba_pool import (
            TpuMambaPool,
            TpuPerLayerState,
        )

        pool = TpuMambaPool.__new__(TpuMambaPool)
        pool.device = "tpu"
        pool.checkpoint_width = 1
        # Multiple layers, both production state dtypes, and three buckets.
        conv = [torch.randn(40, 3, 1024, dtype=torch.bfloat16) for _ in range(2)]
        rec = [torch.randn(40, 4, 128, 128) for _ in range(2)]
        originals = conv + rec
        pool.mamba_cache = TpuPerLayerState(
            conv=[[t.to("tpu") for t in conv]],
            temporal=[t.to("tpu") for t in rec],
        )
        pool.warmup_state_copy(33)
        graphs = counters["stats"]["unique_graphs"]
        for count in (0, 1, 2, 7, 8, 9, 31, 32, 33, 1):
            pool.copy_from(
                torch.ones(count, dtype=torch.long),
                torch.arange(2, 2 + count),
            )
            for state, original in zip(pool._copy_states(), originals):
                actual = state.cpu()
                torch.testing.assert_close(actual[0:2], original[0:2], atol=0, rtol=0)
                if count:
                    torch.testing.assert_close(
                        actual[2 : 2 + count],
                        original[1].expand(count, *original.shape[1:]),
                        atol=0,
                        rtol=0,
                    )
                torch.testing.assert_close(actual[35:], original[35:], atol=0, rtol=0)
        self.assertEqual(graphs, counters["stats"]["unique_graphs"])

    def test_copy_skips_padding_and_large_slots(self):
        from sglang_tpu.mamba_state_copy import compiled_state_copy

        # 32 MiB per slot exceeds VMEM on supported smaller TPU generations.
        original = torch.randn(4, 128, 256, 256)
        state = original.to("tpu")
        # Poison padding with valid, distinct slots. Processing it would overwrite
        # slot 0, so this also tests count=0 warmup without relying on self-copies.
        src = torch.tensor([1] + [2] * 7, dtype=torch.int32, device="tpu")
        dst = torch.tensor([3] + [0] * 7, dtype=torch.int32, device="tpu")
        for count in (0, 1):
            compiled_state_copy()(
                [state],
                src,
                dst,
                torch.tensor([count], dtype=torch.int32, device="tpu"),
            )
            expected = original.clone()
            if count:
                expected[3] = original[1]
            torch.testing.assert_close(state.cpu(), expected, atol=0, rtol=0)

    def test_allocator_ids_stay_on_host_through_cow_collection(self):
        from sglang.srt.configs.mamba_utils import (
            Mamba2CacheParams,
            Mamba2StateDType,
            Mamba2StateShape,
        )
        from sglang.srt.managers.schedule_batch import ReqKvInfo, ScheduleBatch
        from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
        from sglang.srt.platforms import current_platform
        from sglang_tpu.mamba_pool import TpuMambaPool

        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=512,
            n_groups=2,
            num_heads=4,
            head_dim=128,
            state_size=128,
            conv_kernel=4,
        )
        with patch.object(
            current_platform, "get_mamba_pool_cls", return_value=TpuMambaPool
        ):
            pool = HybridReqToTokenPool(
                size=4,
                mamba_size=16,
                mamba_spec_state_size=0,
                max_context_len=256,
                device="tpu",
                enable_memory_saver=False,
                cache_params=Mamba2CacheParams(
                    dtype=Mamba2StateDType(conv=torch.float32, temporal=torch.float32),
                    layers=[0],
                    shape=shape,
                ),
                mamba_layer_ids=[0],
                enable_mamba_extra_buffer=True,
            )
        req = SimpleNamespace(kv=ReqKvInfo(), rid="copy-test")
        pool.alloc([req])
        row = req.kv.req_pool_idx
        self.assertEqual(req.kv.mamba_pool_idx.device.type, "cpu")
        self.assertEqual(req.kv.mamba_ping_pong_track_buffer.device.type, "cpu")
        torch.testing.assert_close(
            pool.req_index_to_mamba_index_mapping[row].cpu(),
            req.kv.mamba_pool_idx.to(torch.int32),
        )
        new_slot = pool.mamba_allocator.alloc(1)
        checkpoint = pool.donate_mamba_ping_pong_slot(req, new_slot)
        self.assertEqual(checkpoint.device.type, "cpu")
        req.kv.mamba_cow_src_index = checkpoint
        batch = SimpleNamespace()
        ScheduleBatch._collect_deferred_mamba_cow_and_clear(batch, [req])
        self.assertEqual(batch.mamba_cow_src_indices.device.type, "cpu")
        self.assertEqual(batch.mamba_cow_dst_indices.device.type, "cpu")
        pool.copy_mamba_state(batch.mamba_cow_src_indices, batch.mamba_cow_dst_indices)
        pool.free_mamba_cache(req)
        pool.mamba_allocator.free(checkpoint)
        self.assertEqual(pool.mamba_allocator.available_size(), 16)

    def test_tracking_upload_has_no_variable_shape_xla_compiles(self):
        from sglang_tpu.compilation.compiled_forward_shim import (
            _pad_mamba_tracking,
            _restore_forward_batch_inputs,
        )

        seq_lens = torch.zeros(4, dtype=torch.int32, device="tpu")

        def prepare(count, prefill):
            batch = SimpleNamespace(
                seq_lens=seq_lens,
                mamba_track_indices=torch.arange(1, count + 1, dtype=torch.int64),
                mamba_track_mask=torch.tensor([True] * count, dtype=torch.bool),
                mamba_track_seqlens=(
                    torch.full((count,), 64, dtype=torch.int64) if prefill else None
                ),
            )
            originals = {
                name: value
                for name, value in vars(batch).items()
                if name.startswith("mamba_track_")
            }
            _pad_mamba_tracking(batch, 4)
            # Repeated preparation must not upload or pad again.
            _pad_mamba_tracking(batch, 4)
            self.assertEqual(batch.mamba_track_indices.device.type, "tpu")
            self.assertEqual(
                batch.mamba_track_indices.cpu().tolist(),
                list(range(1, count + 1)) + [0] * (4 - count),
            )
            self.assertEqual(
                batch.mamba_track_mask.cpu().tolist(),
                [True] * count + [False] * (4 - count),
            )
            if prefill:
                self.assertEqual(
                    batch.mamba_track_seqlens.cpu().tolist(),
                    [64] * count + [0] * (4 - count),
                )
            else:
                self.assertIsNone(batch.mamba_track_seqlens)
            # Isolate tracking restoration from unrelated sequence-length
            # unpadding, which retains its existing device implementation.
            batch.seq_lens = None
            _restore_forward_batch_inputs(batch, count, count)
            for name, value in originals.items():
                self.assertIs(getattr(batch, name), value)

        prepare(4, True)
        torch.accelerator.synchronize()
        misses = torch.tpu._get_cache_misses()
        for prefill in (False, True):
            for count in (0, 1, 2, 3, 4, 1):
                prepare(count, prefill)
        torch.accelerator.synchronize()
        self.assertEqual(torch.tpu._get_cache_misses(), misses)

    def test_final_decode_and_disabled_snapshots(self):
        for length, decode in [(64, False), (1, True)]:
            with self.subTest(length=length):
                states = self.states()
                self.run_op(states, 0, length, 1, track=(2, length), decode=decode)
                for state in states:
                    torch.testing.assert_close(
                        state[2].cpu(), state[1].cpu(), atol=0, rtol=0
                    )
                saved = [t[2].cpu().clone() for t in states]
                self.run_op(states, 0, length, 1, track=(2, -1), decode=decode)
                for state, expected in zip(states, saved):
                    torch.testing.assert_close(state[2].cpu(), expected, atol=0, rtol=0)

    def test_mixed_batch_snapshots(self):
        # Exercise PER_SEQ's rotation past a leading decode and an inert tail.
        full, reference = self.states(), self.states()

        def ids(x):
            return torch.tensor(x, dtype=torch.int32, device="tpu")

        self.op(
            *(t[:130] for t in self.inputs),
            *full,
            *self.weights,
            ids([0, 1, 130, 130]),
            ids([1, 2, 0]),
            ids([1, 1, 2]),
            ids([1, 129, 0]),
            self.nk,
            self.nv,
            self.dk,
            self.dv,
            self.k,
            track_indices=ids([3, 4, 0]),
            track_end=ids([1, 129, -1]),
        )
        self.run_op(reference, 1, 129, 2)
        for actual, expected in zip(full, reference):
            torch.testing.assert_close(
                actual[4].cpu(), expected[2].cpu(), atol=2e-5, rtol=2e-5
            )
            torch.testing.assert_close(actual[3].cpu(), actual[1].cpu(), atol=0, rtol=0)
            self.assertEqual(torch.count_nonzero(actual[0].cpu()).item(), 0)

    def test_bucketed_metadata(self):
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )
        from sglang_tpu.attention_gdn_pallas import (
            TpuGDNAttnBackend,
        )

        backend = TpuGDNAttnBackend.__new__(TpuGDNAttnBackend)
        backend._distribution_cache = {}
        backend._mamba_chunk_size = 64
        backend._extra_buffer_enabled = True
        backend._spec_window = 1
        backend.req_to_token_pool = SimpleNamespace(
            get_mamba_indices=lambda x: x,
            translate_mamba_indices=lambda x: x,
        )

        def t(x, dtype=torch.int64):
            return torch.tensor(x, dtype=dtype, device="tpu")

        batch = SimpleNamespace(
            _tpu_real_b=2,
            forward_mode=ForwardMode.EXTEND,
            seq_lens=t([257, 193, 0, 0]),
            req_pool_indices=t([1, 2, 0, 0]),
            extend_seq_lens=t([193, 129, 0, 0]),
            extend_prefix_lens=t([64, 64, 0, 0]),
            mamba_track_indices=t([3, 4, 0, 0]),
            mamba_track_mask=t([True, True, False, False], torch.bool),
            # First row forces an intermediate branch at absolute depth 192.
            mamba_track_seqlens=t([193, 193, 0, 0]),
        )
        batch.mamba_track_aligned_lens = lambda: ForwardBatch.mamba_track_aligned_lens(
            batch
        )
        with patch(
            "sglang.srt.model_executor.forward_batch_info.mamba_cache_chunk_size",
            return_value=64,
        ):
            backend.init_forward_metadata(batch)
        self.assertEqual(
            backend._query_start_loc.cpu().tolist(), [0, 193, 322, 322, 322]
        )
        self.assertEqual(backend._track_end.cpu().tolist(), [128, 321, -1, -1])
        batch.mamba_track_mask = None
        backend.init_forward_metadata(batch)
        self.assertEqual(backend._track_end.cpu().tolist(), [-1] * 4)


if __name__ == "__main__":
    unittest.main()
