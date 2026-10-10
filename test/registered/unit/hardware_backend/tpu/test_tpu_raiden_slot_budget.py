"""Unit tests for raiden host staging slot sizing.

Pure arithmetic over tensor-shaped stubs, so these run anywhere -- no TPU /
torch_tpu / tpu_raiden required.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang_tpu.disaggregation.raiden.host_pool import (
    node_worker_rank,
    ranks_per_node,
    slot_host_bytes,
    slots_within_budget,
)

GB = 1024**3


class _Buffer:
    """Stand-in for one layer's fused KV tensor ([num_blocks, page, ...])."""

    def __init__(self, shape, itemsize=2):
        self.shape = shape
        self._itemsize = itemsize

    def element_size(self):
        return self._itemsize


def _pool(num_layers, page_size=128, kv_heads=1, head_dim=128):
    # One rank's fused K+V pool: [num_blocks, page_size, 2, kv_heads, head_dim].
    return [
        _Buffer((4096, page_size, 2, kv_heads, head_dim)) for _ in range(num_layers)
    ]


class TestSlotHostBytes(unittest.TestCase):
    def test_counts_every_layer(self):
        page = 128 * 2 * 1 * 128 * 2
        self.assertEqual(slot_host_bytes(_pool(1), 64), page * 64)
        self.assertEqual(slot_host_bytes(_pool(48), 64), page * 48 * 64)

    def test_scales_with_max_blocks(self):
        small = slot_host_bytes(_pool(48), 64)
        large = slot_host_bytes(_pool(48), 2048)
        self.assertEqual(large, small * 32)

    def test_uses_the_widest_layer(self):
        mixed = _pool(2) + [_Buffer((4096, 128, 2, 16, 128))]
        widest = 128 * 2 * 16 * 128 * 2
        self.assertEqual(slot_host_bytes(mixed, 8), widest * 3 * 8)


class TestRanksPerNode(unittest.TestCase):
    def test_single_node_holds_every_rank(self):
        self.assertEqual(ranks_per_node(8, 1, 1, 1), 8)

    def test_ranks_spread_over_nodes(self):
        # TP8 over two four-rank hosts: each host seats four.
        self.assertEqual(ranks_per_node(8, 1, 1, 2), 4)

    def test_attention_dp_replicas_share_the_node(self):
        # attn TP4 x attn DP2 on one host: eight ranks; over two hosts, four.
        self.assertEqual(ranks_per_node(4, 2, 1, 1), 8)
        self.assertEqual(ranks_per_node(4, 2, 1, 2), 4)

    def test_system_dp_replicas_share_the_node(self):
        self.assertEqual(ranks_per_node(4, 1, 2, 1), 8)

    def test_two_node_budget_is_not_split_twice(self):
        # 128 GiB per node, TP8 over two hosts, 20 GiB slots: four local
        # ranks each get 32 GiB and a slot fits; the job-wide rank count
        # would have left 16 GiB and rejected it.
        per_rank = 128 * GB / ranks_per_node(8, 1, 1, 2)
        self.assertEqual(slots_within_budget(per_rank, 20 * GB, 64), 1)
        with self.assertRaises(ValueError):
            slots_within_budget(128 * GB / (8 * 1), 20 * GB, 64)


class TestNodeWorkerRank(unittest.TestCase):
    def test_system_dp_replicas_get_distinct_ranks(self):
        # Two system-DP replicas of TP4 on one host: same TP rank, different
        # replica, so ports (base + 4*rank) and uuid salt never collide.
        first = node_worker_rank(0, 4, 0, 1, system_dp_rank=0)
        second = node_worker_rank(0, 4, 0, 1, system_dp_rank=1)
        self.assertNotEqual(first, second)
        self.assertEqual(second, 4)

    def test_attention_dp_replicas_get_distinct_ranks(self):
        ranks = {
            node_worker_rank(tp, 4, dp, 2, 0) for tp in range(4) for dp in range(2)
        }
        self.assertEqual(ranks, set(range(8)))

    def test_single_replica_is_the_tp_rank(self):
        self.assertEqual(node_worker_rank(3, 8, 0, 1, 0), 3)

    def test_context_parallel_replicas_get_distinct_ranks(self):
        ranks = {
            node_worker_rank(tp, 4, 0, 1, 0, cp, 2)
            for tp in range(4)
            for cp in range(2)
        }
        self.assertEqual(ranks, set(range(8)))

    def test_context_parallel_ranks_share_the_node_budget(self):
        self.assertEqual(ranks_per_node(4, 1, 1, 1, attn_cp_size=2), 8)


class TestSlotsWithinBudget(unittest.TestCase):
    def test_divides_budget_by_slot_cost(self):
        self.assertEqual(slots_within_budget(16 * GB, 1 * GB, 64), 16)

    def test_clamps_to_the_ceiling(self):
        self.assertEqual(slots_within_budget(16 * GB, 1 * GB, 4), 4)

    def test_exact_fit_is_one_slot(self):
        self.assertEqual(slots_within_budget(8 * GB, 8 * GB, 64), 1)

    def test_rejects_a_slot_larger_than_the_budget(self):
        with self.assertRaises(ValueError) as ctx:
            slots_within_budget(1 * GB, 8 * GB, 64)
        self.assertIn("8.00 GiB", str(ctx.exception))
        self.assertIn("1.00 GiB", str(ctx.exception))
        self.assertIn("SGLANG_RAIDEN_HOST_POOL_GB", str(ctx.exception))

    def test_long_context_lowers_the_count_instead_of_overcommitting(self):
        # 8k context at page 128 is 64 blocks/slot; 256k context is 2048.
        budget = 16 * GB
        short = slots_within_budget(budget, slot_host_bytes(_pool(48), 64), 64)
        long = slots_within_budget(budget, slot_host_bytes(_pool(48), 2048), 64)
        self.assertEqual(short, 64)
        self.assertLess(long, short)
        self.assertGreaterEqual(long, 1)


if __name__ == "__main__":
    unittest.main()
