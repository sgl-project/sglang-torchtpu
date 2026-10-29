"""The vendored Kimi-K3 programs on sglang's pools: the programs' own binders
accept the pools sglang_bridge builds, every KDA layer finds its own state
block, and KV pages map to logical blocks."""

import ast
import pathlib
import unittest

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.k3_programs import sglang_bridge as bridge
from sglang_tpu.k3_programs.kimi_k3 import pools
from sglang_tpu.k3_programs.kimi_k3.layout import KDA, MLA, NUM_LAYERS

PACKAGE = pathlib.Path(bridge.__file__).parent


def _plan(num_blocks=40, slots=4):
    return bridge.UnifiedPoolPlan(num_blocks, slots, pools.cache_layout("auto"))


def _kda_groups(groups, kda_names):
    """(group, block size) per kernel KDA layer, as KimiDecoder.bind_kv_caches builds it."""
    of = {name: g for g, group in enumerate(groups) for name in group.layer_names}
    return [(of[name], int(groups[of[name]].block_size)) for name in kda_names]


class TestVendoredPrograms(CustomTestCase):
    def test_no_vllm_imports(self):
        for path in PACKAGE.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else []
                )
                for name in names:
                    with self.subTest(file=str(path.relative_to(PACKAGE)), name=name):
                        self.assertFalse(
                            name == "vllm"
                            or name.startswith(("vllm.", "vllm_torchtpu"))
                        )


class TestUnifiedPools(CustomTestCase):
    def setUp(self):
        self.plan = _plan()
        self.caches = bridge.allocate_pools(self.plan, "cpu")
        (self.raw, self.mla_raw, self.kda_raw, self.mla_names, self.kda_names) = (
            pools.bind_raw_pools(self.caches)
        )
        self.groups = bridge.kv_groups(self.plan, context_len=1 << 20, table_width=8)

    def test_programs_bind_the_pools(self):
        self.assertEqual(len(self.raw), len(MLA))
        self.assertEqual(sorted(self.mla_raw), list(range(len(MLA))))
        self.assertEqual(
            self.kda_raw, [k // bridge.KDA_PER_POOL for k in range(len(KDA))]
        )
        self.assertEqual(tuple(self.raw[0].shape), (40, 128, 2, 640))
        self.assertEqual(bridge.layers_covered(self.groups), set(range(NUM_LAYERS)))

    def test_no_two_layers_of_a_group_share_a_pool(self):
        # Two KDA layers of one group in one pool would write the same state block.
        for g, group in enumerate(self.groups):
            with self.subTest(group=g):
                ptrs = [self.caches[name].data_ptr() for name in group.layer_names]
                self.assertEqual(len(set(ptrs)), len(ptrs))

    def test_every_kda_layer_imports_its_own_state_block(self):
        kda_groups = _kda_groups(self.groups, self.kda_names)
        for slot in (1, 3):
            # Every token on sglang KV page 1 (the slots need not be distinct here).
            tables = bridge.block_tables(
                np.full(300_000, 8192), 300_000, slot, self.plan, table_width=64
            )
            got = pools.kda_blocks_at_length(tables, kda_groups, computed := 300_000)
            want = [
                self.plan.state_block(slot, k % bridge.KDA_PER_POOL)
                for k in range(len(KDA))
            ]
            with self.subTest(slot=slot, computed=computed):
                self.assertEqual(got.tolist(), want)
        # Each (pool, block) holds one layer's state for one slot.
        seen = {}
        for slot in range(1, self.plan.state_slots + 1):
            for k in range(len(KDA)):
                key = (self.kda_raw[k], self.plan.state_block(slot, k % 3))
                self.assertNotIn(key, seen)
                seen[key] = (slot, k)

    def test_state_and_kv_blocks_are_disjoint(self):
        state = {
            self.plan.state_block(s, g)
            for s in range(1, self.plan.state_slots + 1)
            for g in range(3)
        }
        kv = {self.plan.kv_block(p) for p in range(1, self.plan.kv_blocks + 1)}
        self.assertEqual(len(state), 3 * self.plan.state_slots)
        self.assertFalse(state & kv)
        self.assertEqual(max(kv), self.plan.num_blocks - 1)
        self.assertNotIn(0, state | kv)


class TestMlaTable(CustomTestCase):
    def test_pages_map_to_logical_blocks(self):
        plan = _plan()
        step = plan.logical_block_tokens
        self.assertEqual(step, 8192)
        # sglang pages 5, 2, 7 hold the first, second and third logical block.
        row = np.concatenate([np.arange(p * step, (p + 1) * step) for p in (5, 2, 7)])
        table = bridge.mla_table_row(row, 2 * step + 1, plan, width=4)
        self.assertEqual(
            table.tolist(), [plan.kv_block(5), plan.kv_block(2), plan.kv_block(7), 0]
        )
        with self.assertRaisesRegex(ValueError, "the table holds 2"):
            bridge.mla_table_row(row, 2 * step + 1, plan, width=2)
        with self.assertRaisesRegex(ValueError, "does not cover"):
            bridge.mla_table_row(row[:step], 2 * step + 1, plan, width=4)

    def test_fp8_blocks_hold_twice_the_tokens(self):
        plan = bridge.UnifiedPoolPlan(40, 4, pools.cache_layout("fp8"))
        self.assertEqual(plan.logical_block_tokens, 16384)

    def test_plan_sizes_from_bytes(self):
        block = 128 * 2 * 640 * 2
        plan = bridge.plan_pools(block * len(MLA) * 50, state_slots=8)
        self.assertEqual(
            (plan.num_blocks, plan.state_blocks, plan.kv_blocks), (50, 24, 25)
        )
        with self.assertRaisesRegex(ValueError, "cannot hold"):
            bridge.plan_pools(block * len(MLA) * 20, state_slots=8)


if __name__ == "__main__":
    unittest.main()
