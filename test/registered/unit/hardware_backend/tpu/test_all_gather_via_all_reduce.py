"""CPU contract for the all-reduce-based TP all-gather (sgl-project/sglang-torchtpu#1700).

The TPU hooks rebuild the LogitsProcessor logits gathers from the in-place
all-reduce while Dynamo traces a graph. These tests simulate a TP group on CPU
and check that the result equals the concatenation a correct all-gather writes
(bit-identical except that -0.0 becomes +0.0), and that eager calls and the
direct tracer keep the native collective.
"""

import unittest
from types import SimpleNamespace
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


def _parallel(group):
    return SimpleNamespace(tp_group=group, attn_tp_group=group)


class TestLogitsGatherHooks(CustomTestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

    def test_tp_logits_gather_under_dynamo_uses_the_all_reduce(self):
        blocks = _blocks(8, (4, 64), torch.bfloat16)
        group = _SimulatedGroup(blocks, 5)
        fallback = mock.Mock()
        gatherer = agar.TpuLogitsGatherer(fallback)

        with mock.patch.object(agar, "get_parallel", return_value=_parallel(group)):
            got = torch.compile(backend="eager")(gatherer)(blocks[5])

        fallback.assert_not_called()
        self.assertTrue(torch.equal(got, torch.cat(blocks, dim=-1)))

    def test_tp_logits_gather_outside_dynamo_uses_the_wrapped_gatherer(self):
        group = mock.Mock(world_size=8)
        fallback = mock.Mock(return_value="native")
        gatherer = agar.TpuLogitsGatherer(fallback)
        logits = torch.ones(2, 4)
        cases = {"eager": False, "direct tracer": True}
        with mock.patch.object(agar, "get_parallel", return_value=_parallel(group)):
            for name, direct_flag in cases.items():
                with self.subTest(name), mock.patch.object(
                    torch.compiler, "_is_compiling_flag", direct_flag
                ):
                    self.assertEqual(gatherer(logits), "native")
        self.assertEqual(fallback.call_count, 2)
        group.all_reduce.assert_not_called()

    def test_attn_tp_logits_gather_under_dynamo_uses_the_all_reduce(self):
        blocks = _blocks(4, (3, 32), torch.bfloat16)
        group = _SimulatedGroup(blocks, 2)
        processor = SimpleNamespace(vocab_size=128)
        original = mock.Mock()

        def traced(logits):
            return agar.gather_attn_tp_logits_around(original, processor, logits)

        with mock.patch.object(agar, "get_parallel", return_value=_parallel(group)):
            got = torch.compile(backend="eager")(traced)(blocks[2])

        original.assert_not_called()
        self.assertTrue(torch.equal(got, torch.cat(blocks, dim=-1)))

    def test_attn_tp_logits_gather_falls_back_when_not_applicable(self):
        logits = torch.ones(2, 32)
        cases = [
            ("eager", 4, 128, logits, False),
            ("single rank", 1, 128, logits, True),
            ("zero rows", 4, 128, torch.ones(0, 32), True),
            ("uneven vocab", 4, 130, logits, True),
        ]
        for name, world, vocab, rows, dynamo in cases:
            with self.subTest(name):
                group = mock.Mock(world_size=world)
                original = mock.Mock(return_value="native")
                processor = SimpleNamespace(vocab_size=vocab)
                with mock.patch.object(
                    agar, "get_parallel", return_value=_parallel(group)
                ), mock.patch.object(
                    torch.compiler, "is_dynamo_compiling", return_value=dynamo
                ):
                    got = agar.gather_attn_tp_logits_around(original, processor, rows)
                self.assertEqual(got, "native")
                original.assert_called_once_with(processor, rows)
                group.all_reduce.assert_not_called()

    def test_init_hook_wraps_the_logits_gatherer_once(self):
        inner = mock.Mock()
        processor = SimpleNamespace(_logits_gatherer=inner)

        agar.wrap_logits_gatherer(None, processor)
        agar.wrap_logits_gatherer(None, processor)

        self.assertIsInstance(processor._logits_gatherer, agar.TpuLogitsGatherer)
        self.assertIs(processor._logits_gatherer._fallback, inner)


class TestHookRegistration(CustomTestCase):
    def test_hooks_target_logits_gathers_and_leave_collectives_native(self):
        from sglang_tpu.hooks import HOOKS

        by_target = {entry[0]: entry[1] for entry in HOOKS}
        logits = "sglang.srt.layers.logits_processor.LogitsProcessor."
        self.assertIs(by_target[logits + "__init__"], agar.wrap_logits_gatherer)
        self.assertIs(
            by_target[logits + "_gather_attn_tp_logits"],
            agar.gather_attn_tp_logits_around,
        )
        # Per-layer gathers (DP-attention, fused MoE) go through these; they
        # must keep the native collective (#2013).
        coordinator = "sglang.srt.distributed.parallel_state.GroupCoordinator."
        for name in ("all_gather", "all_gather_into_tensor"):
            self.assertNotIn(coordinator + name, by_target)


if __name__ == "__main__":
    unittest.main()
