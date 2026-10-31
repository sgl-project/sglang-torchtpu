"""CPU contract for the all-reduce-based TP all-gather (sgl-project/sglang-torchtpu#1700).

The TPU hook rebuilds GroupCoordinator.all_gather_into_tensor from the in-place
all-reduce while a graph is traced. These tests simulate a TP group on CPU and
check that the result equals the concatenation a correct all-gather writes
(bit-identical except that -0.0 becomes +0.0), and that eager calls keep the
native collective.
"""

import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import all_gather_via_all_reduce as agar

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _SimulatedGroup:
    """One rank of a simulated TP group whose all_reduce sums every rank's buffer.

    Each rank's contribution is the zero-padded buffer the helper would build on
    that rank, so the sum is what a real all-reduce returns to every rank.
    """

    def __init__(self, blocks: list[torch.Tensor], rank: int):
        self._blocks = blocks
        self.world_size = len(blocks)
        self.rank_in_group = rank
        self.reduced_shapes: list[tuple[int, ...]] = []

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self.reduced_shapes.append(tuple(tensor.shape))
        # This rank wrote only its own slot; the other slots are still zero.
        for other in range(self.world_size):
            if other != self.rank_in_group:
                assert torch.count_nonzero(tensor[other]) == 0
        total = torch.zeros_like(tensor)
        for other, block in enumerate(self._blocks):
            contribution = torch.zeros_like(tensor)
            contribution[other] = block
            total += contribution
        return total


def _blocks(world_size: int, shape: tuple[int, ...], dtype: torch.dtype):
    generator = torch.Generator().manual_seed(world_size * 1000 + shape[-1])
    return [
        (torch.randn(shape, generator=generator) * 8).to(dtype)
        for _ in range(world_size)
    ]


class TestGatherBlocksViaAllReduce(CustomTestCase):
    def test_matches_concatenation_bit_exactly(self):
        # Shard widths include 18992 (Qwen3 vocabulary 151936 / 8) and 37984
        # (/ 4), neither a multiple of 128.
        cases = [
            (2, (8, 75968), torch.bfloat16),
            (4, (8, 37984), torch.bfloat16),
            (8, (8, 18992), torch.bfloat16),
            (8, (32, 18992), torch.float32),
            (8, (3, 5, 7), torch.float16),
        ]
        for world_size, shape, dtype in cases:
            blocks = _blocks(world_size, shape, dtype)
            expected = torch.cat(blocks, dim=0)
            for rank in range(world_size):
                with self.subTest(
                    world=world_size, shape=shape, dtype=dtype, rank=rank
                ):
                    group = _SimulatedGroup(blocks, rank)
                    got = agar.gather_blocks_via_all_reduce(group, blocks[rank])
                    self.assertEqual(got.dtype, dtype)
                    self.assertEqual(
                        tuple(got.shape), (world_size * shape[0], *shape[1:])
                    )
                    self.assertTrue(torch.equal(got, expected))
                    self.assertEqual(group.reduced_shapes, [(world_size, *shape)])

    def test_preserves_non_finite_values(self):
        blocks = [
            torch.tensor([[float("inf"), -1.0]]),
            torch.tensor([[float("-inf"), 2.0]]),
        ]
        got = agar.gather_blocks_via_all_reduce(_SimulatedGroup(blocks, 1), blocks[1])
        self.assertTrue(
            torch.equal(got, torch.tensor([[float("inf"), -1.0], [float("-inf"), 2.0]]))
        )

    def test_negative_zero_keeps_its_value_but_not_its_sign_bit(self):
        blocks = [torch.tensor([[-0.0, 1.0]]), torch.tensor([[3.0, -0.0]])]

        got = agar.gather_blocks_via_all_reduce(_SimulatedGroup(blocks, 0), blocks[0])

        self.assertEqual(got.tolist(), [[0.0, 1.0], [3.0, 0.0]])
        self.assertFalse(torch.signbit(got).any())


class TestAllGatherIntoTensorAround(CustomTestCase):
    def test_compiled_call_writes_the_gather_into_output(self):
        blocks = _blocks(8, (8, 18992), torch.bfloat16)
        group = _SimulatedGroup(blocks, 3)
        output = torch.empty(64, 18992, dtype=torch.bfloat16)
        original = mock.Mock()
        with mock.patch.object(torch.compiler, "is_compiling", return_value=True):
            agar.all_gather_into_tensor_around(original, group, output, blocks[3])
        original.assert_not_called()
        self.assertTrue(torch.equal(output, torch.cat(blocks, dim=0)))

    def test_eager_call_uses_the_native_collective(self):
        group = mock.Mock(world_size=8)
        output = torch.empty(16, 4)
        block = torch.ones(2, 4)
        original = mock.Mock(return_value=None)
        with mock.patch.object(torch.compiler, "is_compiling", return_value=False):
            agar.all_gather_into_tensor_around(original, group, output, block)
        original.assert_called_once_with(group, output, block)
        group.all_reduce.assert_not_called()

    def test_single_rank_group_uses_the_native_collective(self):
        group = mock.Mock(world_size=1)
        output = torch.empty(2, 4)
        block = torch.ones(2, 4)
        original = mock.Mock(return_value=None)
        with mock.patch.object(torch.compiler, "is_compiling", return_value=True):
            agar.all_gather_into_tensor_around(original, group, output, block)
        original.assert_called_once_with(group, output, block)


class TestHookRegistration(CustomTestCase):
    def test_hook_targets_group_coordinator_all_gather_into_tensor(self):
        from sglang_tpu.hooks import HOOKS

        target = (
            "sglang.srt.distributed.parallel_state."
            "GroupCoordinator.all_gather_into_tensor"
        )
        entries = [entry for entry in HOOKS if entry[0] == target]
        self.assertEqual(len(entries), 1)
        self.assertIs(entries[0][1], agar.all_gather_into_tensor_around)


if __name__ == "__main__":
    unittest.main()
