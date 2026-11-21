# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for TPU DP-encoder gather (sglang_tpu/hooks/dp_encoder_gather.py)."""

import contextlib
import functools
import inspect
import pkgutil
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.multimodal import mm_utils
from sglang.srt.plugins.hook_registry import HookType
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.hooks import HOOKS
from sglang_tpu.hooks import dp_encoder_gather as hook_module
from sglang_tpu.models import muse_spark_omni

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_TARGET = "sglang.srt.multimodal.mm_utils.gather_dp_encoder_rank_embeddings"
_IN_TREE_GATHER = mm_utils.gather_dp_encoder_rank_embeddings
_TPU_GATHER = functools.partial(hook_module.gather_rank_embeddings_tpu, _IN_TREE_GATHER)


class _SimulatedGroup:
    """One rank's view of an attention-TP group that validates collective operands."""

    def __init__(
        self,
        blocks: list[torch.Tensor],
        rank: int,
        *,
        padded_blocks: list[torch.Tensor] | None = None,
    ):
        self._blocks = blocks
        self._padded_blocks = padded_blocks
        self.rank_in_group = rank
        self.broadcasts: list[tuple[int, tuple[int, ...]]] = []
        self.sent_broadcast_tensors: list[torch.Tensor] = []
        self.all_gathers = 0
        self.last_all_gather_input: torch.Tensor | None = None

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> torch.Tensor:
        self.broadcasts.append((src, tuple(input_.shape)))
        if src == self.rank_in_group:
            self.sent_broadcast_tensors.append(input_)
            assert torch.equal(input_, self._blocks[src])
        else:
            input_.copy_(self._blocks[src])
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == 0
        self.all_gathers += 1
        self.last_all_gather_input = input_
        if self._padded_blocks is not None:
            return torch.cat(self._padded_blocks, dim=0)
        padded = []
        for r, block in enumerate(self._blocks):
            if r == self.rank_in_group:
                padded.append(input_)
            else:
                buffer = torch.full_like(input_, -77.0)
                if block.shape[0] > 0:
                    buffer[: block.shape[0]] = block
                padded.append(buffer)
        return torch.cat(padded, dim=0)


class _Rope2dVisionTower:
    merge_kernel_size = (1, 1)
    config = SimpleNamespace(hidden_size=1)
    device = torch.device("cpu")

    def __call__(self, pixel_values, grid_hw, max_seqlen):
        return pixel_values.reshape(-1, 1, 1)


class _Rope3dVisionTower:
    spatial_merge_size = 2
    out_hidden_size = 3
    device = torch.device("cpu")
    dtype = torch.float32

    def __call__(self, pixel_values, grid_thw):
        merged_rows = pixel_values.shape[0] // 4
        return pixel_values.reshape(merged_rows, 4, 3).sum(dim=1)


def _blocks(
    lengths: list[int],
    row_shape: tuple[int, ...],
    dtype: torch.dtype = torch.bfloat16,
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(sum(lengths) * 31 + len(row_shape) + 7)
    return [
        torch.randn((length, *row_shape), generator=generator, dtype=torch.float32).to(
            dtype
        )
        for length in lengths
    ]


def _gather(fn, blocks: list[torch.Tensor], rank: int):
    group = _SimulatedGroup(blocks, rank)
    lengths = [block.shape[0] for block in blocks]
    with get_parallel().override(
        attn_tp_size=len(blocks), attn_tp_rank=rank, attn_tp_group=group
    ):
        return fn(blocks[rank], lengths), group


class TestDpEncoderGatherHook(CustomTestCase):
    def _assert_standalone_buffer(
        self, tensor: torch.Tensor, *, forbid_ptr: int | None = None
    ) -> None:
        self.assertFalse(tensor._is_view())
        self.assertFalse(tensor.is_neg())
        self.assertFalse(tensor.is_conj())
        self.assertTrue(tensor.is_contiguous())
        self.assertEqual(tensor.storage_offset(), 0)
        self.assertEqual(tensor.untyped_storage().nbytes(), tensor.nbytes)
        if forbid_ptr is not None:
            self.assertNotEqual(tensor.untyped_storage().data_ptr(), forbid_ptr)

    def test_all_gather_mode_matches_core_and_emits_standalone_finite_buffers(self):
        """Every rank must pass a standalone, zero-padded buffer into one all_gather."""
        cases = [
            ([4, 1], (1, 1)),
            ([4784, 0, 9568, 4784, 0, 0, 2048, 1], (8,)),
            ([3, 3, 3, 3], (2, 5)),
            ([0, 5, 3], (6,)),
        ]
        for lengths, row_shape in cases:
            max_len = max(lengths)
            blocks = _blocks(lengths, row_shape)
            for rank, length in enumerate(lengths):
                with self.subTest(lengths=lengths, row_shape=row_shape, rank=rank):
                    expected, ref_group = _gather(_IN_TREE_GATHER, blocks, rank)
                    got, group = _gather(_TPU_GATHER, blocks, rank)

                    self.assertEqual(ref_group.all_gathers, 1)
                    self.assertEqual(group.all_gathers, 1)
                    self.assertEqual(group.broadcasts, [])

                    sent = group.last_all_gather_input
                    self.assertIsNotNone(sent)
                    self.assertEqual(sent.shape, (max_len, *row_shape))
                    self._assert_standalone_buffer(sent)
                    self.assertTrue(torch.isfinite(sent).all())
                    self.assertTrue(torch.equal(sent[:length], blocks[rank]))
                    self.assertTrue(
                        torch.equal(sent[length:], torch.zeros_like(sent[length:]))
                    )

                    non_empty_ptrs = {
                        got[r].untyped_storage().data_ptr()
                        for r, n in enumerate(lengths)
                        if n > 0
                    }
                    for r, n in enumerate(lengths):
                        if n == 0:
                            self.assertEqual(got[r].untyped_storage().nbytes(), 0)
                            self.assertNotIn(
                                got[r].untyped_storage().data_ptr(), non_empty_ptrs
                            )

                    self.assertEqual(len(got), len(lengths))
                    for want, have in zip(expected, got):
                        self.assertEqual(have.shape, want.shape)
                        self.assertEqual(have.dtype, want.dtype)
                        self.assertTrue(torch.equal(have, want))

        # Single non-empty rank uses one isolated broadcast instead of padding
        # empty peers to [tp_size * owner_len, *row_shape].
        single_owner_backing = torch.arange(48, dtype=torch.float32).reshape(8, 6)
        single_owner_blocks = [
            torch.empty((0, 6), dtype=torch.float32),
            single_owner_backing[:5],
            torch.empty((0, 6), dtype=torch.float32),
        ]
        for rank in range(3):
            with self.subTest(single_owner_rank=rank):
                expected, ref_group = _gather(
                    _IN_TREE_GATHER, single_owner_blocks, rank
                )
                got, group = _gather(_TPU_GATHER, single_owner_blocks, rank)
                self.assertEqual(ref_group.all_gathers, 0)
                self.assertEqual(ref_group.broadcasts, [(1, (5, 6))])
                self.assertEqual(group.all_gathers, 0)
                self.assertEqual(group.broadcasts, [(1, (5, 6))])
                if rank == 1:
                    self.assertEqual(len(group.sent_broadcast_tensors), 1)
                    self._assert_standalone_buffer(
                        group.sent_broadcast_tensors[0],
                        forbid_ptr=single_owner_backing.untyped_storage().data_ptr(),
                    )
                for want, have in zip(expected, got):
                    self.assertTrue(torch.equal(have, want))

    def test_materializes_inference_mode_unsafe_squeezed_neg_and_transposed_views(
        self,
    ):
        """Views with _is_view() == False (inference_mode, _unsafe_view) still copy."""
        slice_backing = torch.arange(48, dtype=torch.float32).reshape(8, 6)
        slice_backing[0, 0] = -0.0
        full_slice_view = slice_backing[:4]

        with torch.inference_mode():
            inf_backing = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
            inf_reshape_full = inf_backing.reshape(4, 6)
        self.assertFalse(inf_reshape_full._is_view())

        unsafe_backing = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
        unsafe_full = torch.ops.aten._unsafe_view(unsafe_backing, [4, 6])
        self.assertFalse(unsafe_full._is_view())

        squeezed_full = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6).clone()
        squeezed_full.squeeze_(0)
        self.assertFalse(squeezed_full._is_view())

        neg_full_backing = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        neg_full = torch._neg_view(neg_full_backing)

        neg_partial_backing = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        neg_partial = torch._neg_view(neg_partial_backing)

        transposed_full_backing = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        transposed_full = transposed_full_backing.transpose(0, 1)

        transposed_partial_backing = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        transposed_partial = transposed_partial_backing.transpose(0, 1)

        blocks = [
            full_slice_view,
            inf_reshape_full,
            unsafe_full,
            squeezed_full,
            neg_full,
            neg_partial,
            transposed_full,
            transposed_partial,
        ]
        backing_ptrs = [
            slice_backing.untyped_storage().data_ptr(),
            inf_backing.untyped_storage().data_ptr(),
            unsafe_backing.untyped_storage().data_ptr(),
            squeezed_full.untyped_storage().data_ptr(),
            neg_full_backing.untyped_storage().data_ptr(),
            neg_partial_backing.untyped_storage().data_ptr(),
            transposed_full_backing.untyped_storage().data_ptr(),
            transposed_partial_backing.untyped_storage().data_ptr(),
        ]
        materialized_refs = [
            b.resolve_conj().resolve_neg().contiguous().clone() for b in blocks
        ]
        for rank in range(len(blocks)):
            with self.subTest(rank=rank):
                expected, _ = _gather(_IN_TREE_GATHER, materialized_refs, rank)
                got, group = _gather(_TPU_GATHER, blocks, rank)

                sent = group.last_all_gather_input
                self.assertIsNotNone(sent)
                self._assert_standalone_buffer(sent, forbid_ptr=backing_ptrs[rank])
                for want, have in zip(expected, got):
                    self.assertTrue(torch.equal(have, want))
                    self.assertTrue(
                        torch.equal(torch.signbit(have), torch.signbit(want))
                    )

        # Conjugate complex views resolve conjugation on both full and partial paths.
        conj_backing = torch.complex(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        )
        conj_view = conj_backing.conj()
        for target_rows in (2, 4):
            padded_conj = hook_module._pad_dp_encoder_embeddings_tpu(
                conj_view, target_rows
            )
            self._assert_standalone_buffer(
                padded_conj, forbid_ptr=conj_backing.untyped_storage().data_ptr()
            )
            self.assertTrue(torch.equal(padded_conj[:2], conj_view.resolve_conj()))

    def test_zero_row_and_single_rank_fast_paths_skip_collectives(self):
        """Empty batches, zero-width features, and TP=1 must not issue collectives."""
        self.assertEqual(
            _TPU_GATHER(torch.empty((0, 4), dtype=torch.bfloat16), []),
            [],
        )
        for lengths, row_shape in (
            ([0, 0, 0, 0], (8,)),
            ([3, 1], (0,)),
            ([5], (4,)),
        ):
            blocks = _blocks(lengths, row_shape)
            for rank in range(len(lengths)):
                with self.subTest(lengths=lengths, row_shape=row_shape, rank=rank):
                    got, group = _gather(_TPU_GATHER, blocks, rank)
                    self.assertEqual(group.all_gathers, 0)
                    self.assertEqual(group.broadcasts, [])
                    self.assertEqual(
                        [t.shape for t in got],
                        [(n, *row_shape) for n in lengths],
                    )
                    if lengths == [5]:
                        self.assertTrue(torch.equal(got[0], blocks[0]))

    def test_oversized_local_tensor_is_truncated_to_standalone_buffer(self):
        """A full rank whose local buffer exceeds max_len is sliced and materialized."""
        oversized = torch.arange(36, dtype=torch.float32).reshape(6, 6)
        padded = hook_module._pad_dp_encoder_embeddings_tpu(oversized, 4)
        self.assertEqual(padded.shape, (4, 6))
        self._assert_standalone_buffer(
            padded, forbid_ptr=oversized.untyped_storage().data_ptr()
        )
        self.assertTrue(torch.equal(padded, oversized[:4]))

    def test_broadcast_fallback_mode_materializes_sender_views_and_validates_mode(self):
        """Broadcast mode skips empty ranks and never passes caller views."""
        slice_backing = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        with torch.inference_mode():
            inf_backing = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
            inf_view = inf_backing.reshape(3, 4)

        blocks = [
            slice_backing[:2],
            torch.empty((0, 4), dtype=torch.float32),
            inf_view,
            torch.empty((0, 4), dtype=torch.float32),
        ]
        backing_ptrs = {
            0: slice_backing.untyped_storage().data_ptr(),
            2: inf_backing.untyped_storage().data_ptr(),
        }
        materialized = [b.contiguous().clone() for b in blocks]
        with tpu_envs.SGLANG_TPU_DP_ENCODER_GATHER_MODE.override("broadcast"):
            for rank in range(len(blocks)):
                with self.subTest(rank=rank):
                    expected, _ = _gather(_IN_TREE_GATHER, materialized, rank)
                    got, group = _gather(_TPU_GATHER, blocks, rank)
                    self.assertEqual(group.all_gathers, 0)
                    self.assertEqual(group.broadcasts, [(0, (2, 4)), (2, (3, 4))])
                    if rank in backing_ptrs:
                        self.assertEqual(len(group.sent_broadcast_tensors), 1)
                        self._assert_standalone_buffer(
                            group.sent_broadcast_tensors[0],
                            forbid_ptr=backing_ptrs[rank],
                        )
                    for want, have in zip(expected, got):
                        self.assertTrue(torch.equal(have, want))

        with tpu_envs.SGLANG_TPU_DP_ENCODER_GATHER_MODE.override("invalid_mode"):
            with self.assertRaisesRegex(
                ValueError, "SGLANG_TPU_DP_ENCODER_GATHER_MODE"
            ):
                _gather(_TPU_GATHER, _blocks([0, 0], (4,)), 0)

    def test_rejects_invalid_shapes_and_lengths_before_collectives(self):
        """Invalid tensor rank, negative lengths, or mismatched counts fail on host."""
        with self.assertRaisesRegex(ValueError, "at least 1 dimension"):
            _TPU_GATHER(torch.tensor(1.0), [1])
        with self.assertRaisesRegex(ValueError, "at least 1 dimension"):
            hook_module._pad_dp_encoder_embeddings_tpu(torch.tensor(1.0), 1)
        with self.assertRaisesRegex(ValueError, "at least 1 dimension"):
            hook_module.broadcast_dp_encoder_embedding_tpu(
                torch.tensor(1.0), output_length=1, src=0
            )

        group = _SimulatedGroup(_blocks([2, 2], (4,)), 0)
        with get_parallel().override(
            attn_tp_size=2, attn_tp_rank=0, attn_tp_group=group
        ):
            with self.assertRaisesRegex(ValueError, "non-negative"):
                _TPU_GATHER(torch.zeros((2, 4)), [-1, 2])
            with self.assertRaisesRegex(ValueError, "attn_tp_size"):
                _TPU_GATHER(torch.zeros((2, 4)), [2, 2, 2])
            with self.assertRaisesRegex(ValueError, "row count"):
                _TPU_GATHER(torch.zeros((3, 4)), [2, 2])
            with self.assertRaisesRegex(ValueError, "non-negative"):
                hook_module.broadcast_dp_encoder_embedding_tpu(
                    torch.zeros((2, 4)), output_length=-1, src=0
                )
            with self.assertRaisesRegex(ValueError, "src"):
                hook_module.broadcast_dp_encoder_embedding_tpu(
                    torch.zeros((2, 4)), output_length=2, src=2
                )
            with self.assertRaisesRegex(ValueError, "row count"):
                hook_module.broadcast_dp_encoder_embedding_tpu(
                    torch.zeros((1, 4)), output_length=2, src=0
                )
        self.assertEqual(group.all_gathers, 0)
        self.assertEqual(group.broadcasts, [])

    def test_tpu_device_isolation_syncs_operand_drains_streams_and_forces_defer_never(
        self,
    ):
        """TPU tensors sync operand, drain streams, and force DEFER_NEVER."""
        events: list[tuple] = []
        active_modes: list[str] = ["DEFER_AND_FUSE"]

        class _FakeTpuTensor:
            device = SimpleNamespace(type="tpu")

        fake_exec_mod = types.ModuleType("torch_tpu._internal.execution_mode")
        fake_exec_mod.EagerMode = SimpleNamespace(
            DEFER_NEVER="DEFER_NEVER", DEFER_AND_FUSE="DEFER_AND_FUSE"
        )

        @contextlib.contextmanager
        def fake_set_eager_mode(mode):
            events.append(("enter_mode", mode))
            active_modes.append(mode)
            try:
                yield
            finally:
                active_modes.pop()
                events.append(("exit_mode", active_modes[-1]))

        fake_exec_mod.set_eager_mode = fake_set_eager_mode

        fake_sync_mod = types.ModuleType("torch_tpu._internal.sync.sync")

        def fake_sync(target=None, wait=False):
            events.append(("sync", "none" if target is None else "tensor", wait))

        fake_sync_mod.synchronize = fake_sync

        fake_torch_tpu = types.ModuleType("torch_tpu")
        fake_internal = types.ModuleType("torch_tpu._internal")
        fake_internal.execution_mode = fake_exec_mod
        fake_sync_pkg = types.ModuleType("torch_tpu._internal.sync")
        fake_sync_pkg.sync = fake_sync_mod

        fake_tensor = _FakeTpuTensor()

        class _FakeTpuGroup:
            def all_gather(self, tensor, dim=0):
                events.append(
                    ("all_gather", tensor is fake_tensor, dim, active_modes[-1])
                )
                return "gathered"

        with mock.patch.dict(
            sys.modules,
            {
                "torch_tpu": fake_torch_tpu,
                "torch_tpu._internal": fake_internal,
                "torch_tpu._internal.execution_mode": fake_exec_mod,
                "torch_tpu._internal.sync": fake_sync_pkg,
                "torch_tpu._internal.sync.sync": fake_sync_mod,
            },
        ):
            out = hook_module._isolated_tpu_all_gather(_FakeTpuGroup(), fake_tensor)
            self.assertEqual(out, "gathered")
            self.assertEqual(
                events,
                [
                    ("sync", "tensor", False),
                    ("sync", "none", False),
                    ("enter_mode", "DEFER_NEVER"),
                    ("all_gather", True, 0, "DEFER_NEVER"),
                    ("exit_mode", "DEFER_AND_FUSE"),
                ],
            )

            events.clear()
            out_recv = hook_module._run_isolated_tpu_collective(
                fake_tensor, lambda: "recv", sync_operand=False
            )
            self.assertEqual(out_recv, "recv")
            self.assertEqual(
                events,
                [
                    ("sync", "none", False),
                    ("enter_mode", "DEFER_NEVER"),
                    ("exit_mode", "DEFER_AND_FUSE"),
                ],
            )

    def test_sharded_presharded_and_muse_spark_call_sites_route_through_tpu_helpers(
        self,
    ):
        """All DP-encoder entry points (including Muse len=1) isolate collectives."""
        # 1. run_dp_sharded_mrope_vision_model (uneven [4, 1] on TP2)
        pixel_values = torch.tensor([[1], [2], [3], [4], [7]], dtype=torch.float32)
        sharded_blocks = [
            pixel_values[:4].reshape(-1, 1, 1),
            pixel_values[4:].reshape(-1, 1, 1),
        ]
        for rank in range(2):
            group = _SimulatedGroup(sharded_blocks, rank)
            with (
                mock.patch.object(
                    mm_utils, "gather_dp_encoder_rank_embeddings", _TPU_GATHER
                ),
                get_parallel().override(
                    attn_tp_size=2, attn_tp_rank=rank, attn_tp_group=group
                ),
            ):
                embeddings = mm_utils.run_dp_sharded_mrope_vision_model(
                    _Rope2dVisionTower(),
                    pixel_values,
                    [[2, 2], [1, 1]],
                    rope_type="rope_2d",
                )
            self.assertEqual(group.all_gathers, 1)
            self.assertTrue(torch.equal(embeddings, pixel_values.reshape(-1, 1, 1)))

        # 2. run_dp_presharded_mrope_vision_model (uneven [2, 0, 1, 0] on TP4)
        tower_3d = _Rope3dVisionTower()
        unit_pixels = [
            torch.full((4, 3), 2.0),
            torch.full((4, 3), 3.0),
            torch.full((4, 3), 5.0),
        ]
        sample_counts = [2, 0, 1, 0]
        global_grids = [[1, 2, 2], [1, 2, 2], [1, 2, 2]]
        rank_pixels = [
            torch.cat(unit_pixels[:2], dim=0),
            torch.empty((0, 3), dtype=torch.float32),
            unit_pixels[2],
            torch.empty((0, 3), dtype=torch.float32),
        ]
        rank_grids = [[[1, 2, 2], [1, 2, 2]], [], [[1, 2, 2]], []]
        expected_3d = torch.cat([tower_3d(u, [[1, 2, 2]]) for u in unit_pixels], dim=0)
        expected_rank_blocks = [
            tower_3d(rank_pixels[0], rank_grids[0]),
            torch.empty((0, 3), dtype=torch.float32),
            tower_3d(rank_pixels[2], rank_grids[2]),
            torch.empty((0, 3), dtype=torch.float32),
        ]
        for rank in range(4):
            group = _SimulatedGroup(expected_rank_blocks, rank)
            with (
                mock.patch.object(
                    mm_utils, "gather_dp_encoder_rank_embeddings", _TPU_GATHER
                ),
                get_parallel().override(
                    attn_tp_size=4, attn_tp_rank=rank, attn_tp_group=group
                ),
            ):
                out_3d = mm_utils.run_dp_presharded_mrope_vision_model(
                    tower_3d,
                    rank_pixels[rank],
                    rank_grids[rank],
                    global_grids,
                    sample_counts,
                )
            self.assertEqual(group.all_gathers, 1)
            self.assertTrue(torch.equal(out_3d, expected_3d))

        # 3. muse_spark_omni._run_dp_sharded_encode multi-item ([3, 2, 1, 0] on TP4)
        features = [
            torch.full((3, 4), 10.0),
            torch.full((2, 4), 20.0),
            torch.full((1, 4), 30.0),
        ]
        costs = [12, 8, 4]
        tokens = [3, 2, 1]
        muse_blocks = [
            features[0],
            features[1],
            features[2],
            torch.empty((0, 4), dtype=torch.float32),
        ]
        for rank in range(4):
            group = _SimulatedGroup(muse_blocks, rank)
            with get_parallel().override(
                attn_tp_size=4, attn_tp_rank=rank, attn_tp_group=group
            ):
                out_muse = muse_spark_omni._run_dp_sharded_encode(
                    features=features,
                    encode_cost_per_item=costs,
                    tokens_per_item=tokens,
                    encode_fn=lambda items: torch.cat(items, dim=0),
                    out_hidden_size=4,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
            self.assertEqual(group.all_gathers, 1)
            self.assertTrue(torch.equal(out_muse, torch.cat(features, dim=0)))

        # 4. muse_spark_omni._run_dp_sharded_encode single-item fast path with
        #    non-zero owner rank and inference_mode reshape view from encode_fn.
        single_backing = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)

        def _view_encode(items):
            assert len(items) == 1
            with torch.inference_mode():
                return single_backing.reshape(3, 4)

        single_blocks = [
            torch.empty((0, 4), dtype=torch.float32),
            torch.empty((0, 4), dtype=torch.float32),
            single_backing.reshape(3, 4).clone(),
            torch.empty((0, 4), dtype=torch.float32),
        ]
        for rank in range(4):
            group = _SimulatedGroup(single_blocks, rank)
            with (
                mock.patch.object(
                    muse_spark_omni,
                    "get_dp_encoder_lb_assignment",
                    return_value=([0], [0, 0, 1, 0], [0, 0, 12, 0]),
                ),
                get_parallel().override(
                    attn_tp_size=4, attn_tp_rank=rank, attn_tp_group=group
                ),
            ):
                out_single = muse_spark_omni._run_dp_sharded_encode(
                    features=[torch.ones((3, 4), dtype=torch.float32)],
                    encode_cost_per_item=[12],
                    tokens_per_item=[3],
                    encode_fn=_view_encode,
                    out_hidden_size=4,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
            self.assertEqual(group.all_gathers, 0)
            self.assertEqual(group.broadcasts, [(2, (3, 4))])
            if rank == 2:
                self.assertEqual(len(group.sent_broadcast_tensors), 1)
                self._assert_standalone_buffer(
                    group.sent_broadcast_tensors[0],
                    forbid_ptr=single_backing.untyped_storage().data_ptr(),
                )
            self.assertTrue(torch.equal(out_single, single_backing.reshape(3, 4)))

        # 5. run_dp_sharded_mrope_vision_model single-image path with non-zero
        #    owner rank and inference_mode reshape view from vision_model.
        sharded_single_backing = torch.arange(4, dtype=torch.float32).reshape(
            1, 4, 1, 1
        )

        class _SingleViewRope2dTower(_Rope2dVisionTower):
            def __call__(self, pv, grid_hw, max_seqlen):
                with torch.inference_mode():
                    return sharded_single_backing.reshape(4, 1, 1)

        sharded_single_blocks = [
            torch.empty((0, 1, 1), dtype=torch.float32),
            sharded_single_backing.reshape(4, 1, 1).clone(),
        ]
        for rank in range(2):
            group = _SimulatedGroup(sharded_single_blocks, rank)
            with (
                mock.patch.object(
                    mm_utils, "gather_dp_encoder_rank_embeddings", _TPU_GATHER
                ),
                mock.patch.object(
                    mm_utils,
                    "get_dp_encoder_lb_assignment",
                    return_value=([0], [0, 1], [0, 4]),
                ),
                get_parallel().override(
                    attn_tp_size=2, attn_tp_rank=rank, attn_tp_group=group
                ),
            ):
                out_sharded_single = mm_utils.run_dp_sharded_mrope_vision_model(
                    _SingleViewRope2dTower(),
                    pixel_values[:4],
                    [[2, 2]],
                    rope_type="rope_2d",
                )
            self.assertEqual(group.all_gathers, 0)
            self.assertEqual(group.broadcasts, [(1, (4, 1, 1))])
            if rank == 1:
                self.assertEqual(len(group.sent_broadcast_tensors), 1)
                self._assert_standalone_buffer(
                    group.sent_broadcast_tensors[0],
                    forbid_ptr=sharded_single_backing.untyped_storage().data_ptr(),
                )
            self.assertTrue(torch.equal(out_sharded_single, sharded_single_blocks[1]))

        # 6. muse_spark_omni._run_dp_sharded_encode empty features on TP=1 and TP=4.
        for tp_size in (1, 4):
            with get_parallel().override(
                attn_tp_size=tp_size, attn_tp_rank=0, attn_tp_group=None
            ):
                empty_muse = muse_spark_omni._run_dp_sharded_encode(
                    features=[],
                    encode_cost_per_item=[],
                    tokens_per_item=[],
                    encode_fn=lambda _items: (_ for _ in ()).throw(
                        AssertionError("empty features must not call encode_fn")
                    ),
                    out_hidden_size=4,
                    device=torch.device("cpu"),
                    dtype=torch.bfloat16,
                )
            self.assertEqual(empty_muse.shape, (0, 4))
            self.assertEqual(empty_muse.dtype, torch.bfloat16)


class TestDpEncoderGatherHookRegistration(CustomTestCase):
    def test_registered_around_the_core_seam_with_matching_signature(self):
        entries = [h for h in HOOKS if h[1] is hook_module.gather_rank_embeddings_tpu]
        self.assertEqual(
            entries,
            [(_TARGET, hook_module.gather_rank_embeddings_tpu, HookType.AROUND)],
        )
        self.assertIs(pkgutil.resolve_name(_TARGET), _IN_TREE_GATHER)

        def parameters(fn, skip: int) -> list[tuple]:
            params = list(inspect.signature(fn).parameters.values())[skip:]
            return [(p.name, p.kind, p.default) for p in params]

        self.assertEqual(
            parameters(hook_module.gather_rank_embeddings_tpu, skip=1),
            parameters(_IN_TREE_GATHER, skip=0),
        )


if __name__ == "__main__":
    unittest.main()
