# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for DP-encoder per-rank broadcasts (hooks/dp_encoder_gather.py)."""

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
from sglang_tpu.hooks import HOOKS
from sglang_tpu.hooks import dp_encoder_gather as hook_module

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_TARGET = "sglang.srt.multimodal.mm_utils.gather_dp_encoder_rank_embeddings"
_IN_TREE_GATHER = mm_utils.gather_dp_encoder_rank_embeddings
_TPU_GATHER = functools.partial(hook_module.broadcast_rank_embeddings, _IN_TREE_GATHER)


class _SimulatedGroup:
    """One rank's view of an attention-TP group that validates broadcast operands."""

    def __init__(self, blocks: list[torch.Tensor], rank: int):
        self._blocks = blocks
        self.rank_in_group = rank
        self.broadcasts: list[tuple[int, tuple[int, ...]]] = []
        self.sent_tensors: list[torch.Tensor] = []
        self.all_gathers = 0

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> torch.Tensor:
        self.broadcasts.append((src, tuple(input_.shape)))
        if src == self.rank_in_group:
            self.sent_tensors.append(input_)
            assert torch.equal(input_, self._blocks[src])
        else:
            input_.copy_(self._blocks[src])
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == 0
        self.all_gathers += 1
        padded = []
        for block in self._blocks:
            buffer = torch.full_like(input_, -1.0)
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


def _blocks(lengths: list[int], row_shape: tuple[int, ...]) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(sum(lengths) * 31 + len(row_shape))
    return [
        torch.randn((length, *row_shape), generator=generator) for length in lengths
    ]


def _gather(fn, blocks: list[torch.Tensor], rank: int):
    group = _SimulatedGroup(blocks, rank)
    lengths = [block.shape[0] for block in blocks]
    with get_parallel().override(
        attn_tp_size=len(blocks), attn_tp_rank=rank, attn_tp_group=group
    ):
        return fn(blocks[rank], lengths), group


class TestBroadcastRankEmbeddings(CustomTestCase):
    def test_matches_the_padded_all_gather_on_every_rank(self):
        """Per-rank broadcasts return the same per-rank tensors as padded all_gather."""
        cases = [
            ([4, 1], (1, 1)),
            ([4784, 0, 9568, 4784, 0, 0, 2048, 1], (8,)),
            ([3, 3, 3, 3], (2, 5)),
            ([0, 5, 3], (6,)),
            ([0, 5], (6,)),
        ]
        for lengths, row_shape in cases:
            non_empty = [r for r, n in enumerate(lengths) if n > 0]
            blocks = _blocks(lengths, row_shape)
            for rank in range(len(lengths)):
                with self.subTest(lengths=lengths, row_shape=row_shape, rank=rank):
                    expected, ref_group = _gather(_IN_TREE_GATHER, blocks, rank)
                    got, group = _gather(_TPU_GATHER, blocks, rank)

                    if len(non_empty) == 1:
                        self.assertEqual(ref_group.all_gathers, 0)
                        self.assertEqual(len(ref_group.broadcasts), 1)
                    else:
                        self.assertEqual(ref_group.all_gathers, 1)
                    self.assertEqual(group.all_gathers, 0)
                    self.assertEqual(len(got), len(lengths))
                    for want, have in zip(expected, got):
                        self.assertEqual(have.shape, want.shape)
                        self.assertEqual(have.dtype, want.dtype)
                        self.assertTrue(torch.equal(have, want))

    def test_broadcasts_only_ranks_with_rows_in_rank_order(self):
        blocks = _blocks([2, 0, 3, 0], (4,))
        for rank in range(len(blocks)):
            with self.subTest(rank=rank):
                _, group = _gather(_TPU_GATHER, blocks, rank)
                self.assertEqual(group.broadcasts, [(0, (2, 4)), (2, (3, 4))])

    def test_sender_materializes_views_without_aliasing_caller_storage(self):
        """ProcessGroupTpu::broadcast receives a standalone sender buffer."""
        slice_backing = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        slice_view = slice_backing[:3]

        with torch.inference_mode():
            inf_backing = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
            inf_reshape_view = inf_backing.reshape(4, 4)

        unsafe_backing = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        unsafe_view = torch.ops.aten._unsafe_view(unsafe_backing, [3, 4])

        neg_backing = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        neg_view = torch._neg_view(neg_backing)

        blocks = [slice_view, inf_reshape_view, unsafe_view, neg_view]
        backing_ptrs = [
            slice_backing.untyped_storage().data_ptr(),
            inf_backing.untyped_storage().data_ptr(),
            unsafe_backing.untyped_storage().data_ptr(),
            neg_backing.untyped_storage().data_ptr(),
        ]
        for rank in range(len(blocks)):
            with self.subTest(rank=rank):
                got, group = _gather(_TPU_GATHER, blocks, rank)
                self.assertEqual(len(group.sent_tensors), 1)
                sent = group.sent_tensors[0]
                self.assertFalse(sent._is_view())
                self.assertFalse(sent.is_neg())
                self.assertFalse(sent.is_conj())
                self.assertTrue(sent.is_contiguous())
                self.assertEqual(sent.storage_offset(), 0)
                self.assertEqual(sent.untyped_storage().nbytes(), sent.nbytes)
                self.assertNotEqual(
                    sent.untyped_storage().data_ptr(), backing_ptrs[rank]
                )
                for r, want in enumerate(blocks):
                    self.assertTrue(torch.equal(got[r], want))

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
                    got_core, core_group = _gather(_IN_TREE_GATHER, blocks, rank)
                    self.assertEqual(core_group.all_gathers, 0)
                    self.assertEqual(
                        [t.shape for t in got_core],
                        [(n, *row_shape) for n in lengths],
                    )

    def test_rejects_mismatched_or_negative_lengths_before_collectives(self):
        """Invalid shapes or lengths fail on the host before entering a collective."""
        with self.assertRaisesRegex(ValueError, "at least 1 dimension"):
            _TPU_GATHER(torch.tensor(1.0), [1])

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
        self.assertEqual(group.broadcasts, [])

    def test_tpu_isolation_syncs_sender_drains_streams_and_forces_defer_never(self):
        """TPU broadcasts drain streams and force DEFER_NEVER on every rank."""
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
            hook_module._run_isolated_tpu_collective(
                fake_tensor,
                lambda: events.append(("bcast_send", active_modes[-1])),
                sync_operand=True,
            )
            self.assertEqual(
                events,
                [
                    ("sync", "tensor", False),
                    ("sync", "none", False),
                    ("enter_mode", "DEFER_NEVER"),
                    ("bcast_send", "DEFER_NEVER"),
                    ("exit_mode", "DEFER_AND_FUSE"),
                ],
            )

            events.clear()
            hook_module._run_isolated_tpu_collective(
                fake_tensor,
                lambda: events.append(("bcast_recv", active_modes[-1])),
                sync_operand=False,
            )
            self.assertEqual(
                events,
                [
                    ("sync", "none", False),
                    ("enter_mode", "DEFER_NEVER"),
                    ("bcast_recv", "DEFER_NEVER"),
                    ("exit_mode", "DEFER_AND_FUSE"),
                ],
            )

    def test_sharded_and_presharded_mrope_vision_models_call_the_seam(self):
        pixel_values = torch.tensor([[1], [2], [3], [4], [7]], dtype=torch.float32)
        blocks = [
            pixel_values[:4].reshape(-1, 1, 1),
            pixel_values[4:].reshape(-1, 1, 1),
        ]
        for rank in range(2):
            with self.subTest(call="sharded", rank=rank):
                group = _SimulatedGroup(blocks, rank)
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

                self.assertEqual(group.all_gathers, 0)
                self.assertEqual(group.broadcasts, [(0, (4, 1, 1)), (1, (1, 1, 1))])
                self.assertTrue(torch.equal(embeddings, pixel_values.reshape(-1, 1, 1)))

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
            with self.subTest(call="presharded", rank=rank):
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
                self.assertEqual(group.all_gathers, 0)
                self.assertEqual(group.broadcasts, [(0, (2, 3)), (2, (1, 3))])
                self.assertTrue(torch.equal(out_3d, expected_3d))

        # Single-image sharded call site with non-zero owner rank and view output.
        single_backing = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1)

        class _SingleViewRope2dTower(_Rope2dVisionTower):
            def __call__(self, pv, grid_hw, max_seqlen):
                with torch.inference_mode():
                    return single_backing.reshape(4, 1, 1)

        single_blocks = [
            torch.empty((0, 1, 1), dtype=torch.float32),
            single_backing.reshape(4, 1, 1).clone(),
        ]
        for rank in range(2):
            with self.subTest(call="sharded_single_image", rank=rank):
                group = _SimulatedGroup(single_blocks, rank)
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
                    out_single = mm_utils.run_dp_sharded_mrope_vision_model(
                        _SingleViewRope2dTower(),
                        pixel_values[:4],
                        [[2, 2]],
                        rope_type="rope_2d",
                    )
                self.assertEqual(group.all_gathers, 0)
                self.assertEqual(group.broadcasts, [(1, (4, 1, 1))])
                if rank == 1:
                    self.assertEqual(len(group.sent_tensors), 1)
                    self.assertNotEqual(
                        group.sent_tensors[0].untyped_storage().data_ptr(),
                        single_backing.untyped_storage().data_ptr(),
                    )
                self.assertTrue(torch.equal(out_single, single_blocks[1]))


class TestDpEncoderGatherHookRegistration(CustomTestCase):
    def test_registered_around_the_core_seam_with_matching_signature(self):
        entries = [h for h in HOOKS if h[1] is hook_module.broadcast_rank_embeddings]
        self.assertEqual(
            entries,
            [(_TARGET, hook_module.broadcast_rank_embeddings, HookType.AROUND)],
        )
        self.assertIs(pkgutil.resolve_name(_TARGET), _IN_TREE_GATHER)

        def parameters(fn, skip: int) -> list[tuple]:
            params = list(inspect.signature(fn).parameters.values())[skip:]
            return [(p.name, p.kind, p.default) for p in params]

        self.assertEqual(
            parameters(hook_module.broadcast_rank_embeddings, skip=1),
            parameters(_IN_TREE_GATHER, skip=0),
        )


if __name__ == "__main__":
    unittest.main()
