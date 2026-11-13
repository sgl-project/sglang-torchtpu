"""The vendored Kimi-K3 programs on sglang's pools: the programs' own binders
accept the pools sglang_bridge builds, every KDA layer finds its own state
block, and KV pages map to logical blocks."""

import ast
import pathlib
import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.k3_programs import sglang_bridge as bridge
from sglang_tpu.k3_programs import sglang_config
from sglang_tpu.k3_programs import sglang_step as step
from sglang_tpu.k3_programs.kimi_k3 import pools
from sglang_tpu.k3_programs.kimi_k3 import program as k3_program
from sglang_tpu.k3_programs.kimi_k3.layout import KDA, MLA, NUM_LAYERS

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

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
                np.full(300_000, 8192)[::8192], 300_000, slot, self.plan, table_width=64
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
        # KV page p is block p; the state follows, slot 1 after slot 0's three.
        self.assertEqual(min(state), self.plan.state_base + 3)
        self.assertEqual(max(state), self.plan.num_blocks - 1)
        self.assertEqual(
            [self.plan.kv_block(p) for p in (0, 1, self.plan.kv_blocks)],
            [0, 1, self.plan.kv_blocks],
        )
        self.assertEqual(
            self.plan.state_row(2),
            self.plan.state_block(2, 0),
        )
        self.assertFalse(state & kv)
        self.assertEqual(max(kv), self.plan.kv_blocks)
        self.assertNotIn(0, state | kv)


class TestDraftPools(CustomTestCase):
    def test_draft_pools_follow_the_target_pools(self):
        plan = _plan()
        caches = bridge.allocate_pools(
            plan, "cpu", draft_layers=bridge.DSPARK_DRAFT_LAYERS
        )
        draft = pools.draft_raw_pools(caches)
        self.assertEqual(len(draft), bridge.DSPARK_DRAFT_LAYERS)
        raw, *_ = pools.bind_raw_pools(caches)
        for p in draft:
            self.assertEqual((p.shape, p.dtype), (raw[0].shape, raw[0].dtype))
            self.assertNotIn(p.data_ptr(), {r.data_ptr() for r in raw})
        groups = bridge.kv_groups(
            plan, 1 << 20, 8, draft_layers=bridge.DSPARK_DRAFT_LAYERS
        )
        self.assertEqual(
            bridge.layers_covered(groups),
            set(range(NUM_LAYERS + bridge.DSPARK_DRAFT_LAYERS)),
        )
        self.assertTrue(
            all(bridge.draft_layer_name(i) in groups[0].layer_names for i in range(5))
        )


class TestPrefixTails(CustomTestCase):
    def test_tail_blocks_follow_the_state_and_leave_kv_and_slots_alone(self):
        plain = _plan(num_blocks=60, slots=4)
        tails = bridge.UnifiedPoolPlan(60, 4, pools.cache_layout("auto"), tail_slots=2)
        self.assertEqual(tails.tail_blocks, 8)
        self.assertEqual(tails.kv_blocks, plain.kv_blocks - 8)
        tail = {tails.tail_kv_block(t) for t in range(2)} | {
            tails.tail_state_block(t, g) for t in range(2) for g in range(3)
        }
        kv = set(range(1, tails.kv_blocks + 1))
        state = {tails.state_block(s, g) for s in range(1, 5) for g in range(3)}
        self.assertEqual(len(tail), 8)
        self.assertFalse(tail & kv or tail & state)
        self.assertEqual(max(tail), tails.num_blocks - 1)
        with self.assertRaisesRegex(ValueError, "prefix tail 2"):
            tails.tail_kv_block(2)


class TestTransferBuffers(CustomTestCase):
    """sglang's P/D transfer reaches page p / slot i at ptr + index * item_len."""

    def setUp(self):
        self.plan = _plan()
        self.caches = bridge.allocate_pools(self.plan, "cpu", draft_layers=5)
        self.block = 128 * 2 * 640 * 2

    def _bounds(self, tensor):
        return tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * 2

    def test_kv_pages(self):
        ptrs, lens, items = bridge.kv_buf_infos(self.caches, self.plan)
        tensors = [self.caches[bridge.mla_layer_name(layer)] for layer in MLA]
        tensors += pools.draft_raw_pools(self.caches)
        self.assertEqual(len(ptrs), len(MLA) + 5)
        for ptr, length, item, pool in zip(ptrs, lens, items, tensors):
            self.assertEqual(item, self.block)
            lo, hi = self._bounds(pool)
            self.assertGreaterEqual(ptr, lo)
            self.assertLessEqual(ptr + length, hi)
            for page in (1, 7, self.plan.kv_blocks):
                self.assertEqual(
                    ptr + page * item, lo + self.plan.kv_block(page) * self.block
                )

    def test_state_slots(self):
        ptrs, lens, items = bridge.state_buf_infos(self.caches, self.plan)
        self.assertEqual(len(ptrs), len(KDA))
        for k, (ptr, length, item) in enumerate(zip(ptrs, lens, items)):
            pool = self.caches[bridge.kda_layer_name(KDA[k])]
            lo, hi = self._bounds(pool)
            self.assertEqual(item, 3 * self.block)
            self.assertGreaterEqual(ptr, lo)
            # The state range follows the KV pages to the end of the pool.
            self.assertEqual(ptr, lo + (self.plan.state_base + k % 3) * self.block)
            self.assertEqual(ptr + length, hi)
            for slot in range(1, self.plan.state_slots + 1):
                address = ptr + slot * item
                self.assertEqual(
                    address,
                    lo + self.plan.state_block(slot, k % 3) * self.block,
                )
                self.assertLessEqual(address + self.block, ptr + length)


class TestMlaTable(CustomTestCase):
    def test_pages_map_to_logical_blocks(self):
        plan = _plan()
        step = plan.logical_block_tokens
        self.assertEqual(step, 8192)
        # sglang pages 5, 2, 7 hold the first, second and third logical block.
        row = np.concatenate([np.arange(p * step, (p + 1) * step) for p in (5, 2, 7)])
        table = bridge.mla_table_row(row[::step], 2 * step + 1, plan, width=4)
        self.assertEqual(
            table.tolist(), [plan.kv_block(5), plan.kv_block(2), plan.kv_block(7), 0]
        )
        with self.assertRaisesRegex(ValueError, "the table holds 2"):
            bridge.mla_table_row(row[::step], 2 * step + 1, plan, width=2)
        with self.assertRaisesRegex(ValueError, "does not cover"):
            bridge.mla_table_row(row[:step:step], 2 * step + 1, plan, width=4)
        unallocated = np.concatenate([row, np.zeros(step, row.dtype)])
        with self.assertRaisesRegex(ValueError, r"blocks \[3\] .* not allocated"):
            bridge.mla_table_row(unallocated[::step], 3 * step + 1, plan, width=4)
        # page_starts: one slot per logical block of every row.
        table2d = np.stack([row, row])
        self.assertEqual(bridge.page_starts(table2d, plan).shape, (2, 3))

    def test_fp8_blocks_hold_twice_the_tokens(self):
        plan = bridge.UnifiedPoolPlan(40, 4, pools.cache_layout("fp8"))
        self.assertEqual(plan.logical_block_tokens, 16384)

    def test_plan_sizes_from_bytes(self):
        block = 128 * 2 * 640 * 2
        plan = bridge.plan_pools(block * len(MLA) * 50, state_slots=8)
        self.assertEqual(
            (plan.num_blocks, plan.state_blocks, plan.kv_blocks), (50, 27, 22)
        )
        with self.assertRaisesRegex(ValueError, "cannot hold"):
            bridge.plan_pools(block * len(MLA) * 20, state_slots=8)
        # The draft's five pools take their share of the same bytes.
        plan = bridge.plan_pools(block * (len(MLA) + 5) * 50, 8, draft_layers=5)
        self.assertEqual(plan.num_blocks, 50)


class TestStepConversion(CustomTestCase):
    def test_sampling_follows_sglang_encodings(self):
        greedy = SimpleNamespace(
            temperature=1.0, top_p=1.0, top_k=1, sampling_seed=None
        )
        spec = step.sampling_spec(greedy, "r1")
        self.assertEqual((spec.temperature, spec.top_k), (0.0, 0))
        sampled = SimpleNamespace(
            temperature=0.7, top_p=0.9, top_k=TOP_K_ALL, sampling_seed=5
        )
        self.assertEqual(step.sampling_spec(sampled, "r1").as_tuple(), (0.7, 0.9, 0, 5))
        topk = SimpleNamespace(temperature=1.0, top_p=1.0, top_k=20, sampling_seed=None)
        # Unseeded: every rank derives the same seed from the request id.
        a, b = step.sampling_spec(topk, "r9"), step.sampling_spec(topk, "r9")
        self.assertEqual(a, b)
        self.assertEqual(a.top_k, 20)

    def test_request_input(self):
        params = SimpleNamespace(
            temperature=1.0, top_p=1.0, top_k=1, sampling_seed=None
        )
        tables = (np.zeros(4, np.int32),) * 4
        tokens = np.array([1, 2, 3, 4], np.int32)

        def make(computed, scheduled):
            return step.request_input(
                rid="r",
                token_ids=tokens,
                num_tokens=4,
                num_prompt_tokens=3,
                computed=computed,
                scheduled=scheduled,
                block_tables=tables,
                sampling_params=params,
            )

        req = make(3, 1)
        self.assertEqual((req.end, req.num_tokens, req.num_prompt_tokens), (4, 4, 3))
        self.assertFalse(req.partial)
        # Draft slots past the known tokens.
        drafts = make(3, 8)
        self.assertEqual((drafts.end, drafts.num_tokens), (11, 4))
        with self.assertRaisesRegex(ValueError, "leave none of its 4 known"):
            make(4, 1)

    def test_accepted_tokens(self):
        tokens = np.array(
            [[5, 6, 7, -1], [9, -1, -1, -1], [-1, -1, -1, -1], [1, 1, 1, 1]]
        )
        flat, lens = step.accepted(tokens, num_requests=3)
        self.assertEqual((flat, lens), ([5, 6, 7, 9], [3, 1, 0]))
        with self.assertRaisesRegex(ValueError, "not -1 padded"):
            step.accepted(np.array([[-1, 4]]), num_requests=1)


class TestEngineConfig(CustomTestCase):
    def test_golden_acceptance_lengths_as_rates(self):
        rates = sglang_config.acceptance_length_to_rates
        self.assertEqual(rates(3.78, 7)[:2], (1.0, 1.0))
        self.assertAlmostEqual(rates(3.78, 7)[2], 0.78)
        self.assertEqual(rates(3.78, 7)[3:], (0.0,) * 4)
        self.assertEqual(rates(2.87, 3)[0], 1.0)
        self.assertAlmostEqual(rates(2.87, 3)[1], 0.87)
        self.assertEqual(rates(2.87, 3)[2], 0.0)
        self.assertAlmostEqual(sum(rates(3.78, 7)) + 1, 3.78)
        with self.assertRaisesRegex(ValueError, "in \\[1, 4\\]"):
            rates(4.5, 3)

    def test_configs_pass_the_programs_check(self):
        decode = sglang_config.engine_config(
            role="decode",
            max_running_requests=8,
            context_length=1 << 20,
            chunked_prefill_size=8192,
            speculative_tokens=7,
            simulate_acc_lengths={7: 3.78, 3: 2.87},
        )
        k3_program.check(decode)
        self.assertEqual(
            decode.num_speculative_tokens_per_batch_size, ((1, 4, 7), (5, 8, 3))
        )
        self.assertEqual(
            [w for w, _ in decode.synthetic_acceptance_rates_by_width], [3, 7]
        )
        self.assertEqual(
            (decode.block_size, decode.cp_kv_cache_interleave_size), (256, 256)
        )
        prefill = sglang_config.engine_config(
            role="prefill",
            max_running_requests=16,
            context_length=1 << 20,
            chunked_prefill_size=16384,
            kv_cache_dtype="fp8",
        )
        k3_program.check(prefill)
        self.assertEqual(prefill.block_size, 512)
        # A page-sized fp8 chunk runs in the prefill's 8192-token windows.
        self.assertEqual(prefill.max_num_batched_tokens, 8192)

    def test_every_scheduled_width_needs_a_length(self):
        with self.assertRaisesRegex(
            ValueError, "no simulated acceptance length for \\[3\\]"
        ):
            sglang_config.engine_config(
                role="decode",
                max_running_requests=8,
                context_length=1 << 20,
                chunked_prefill_size=8192,
                speculative_tokens=7,
                simulate_acc_lengths={7: 3.78},
            )
        with self.assertRaisesRegex(ValueError, "roles"):
            sglang_config.engine_config(
                role="both",
                max_running_requests=1,
                context_length=1,
                chunked_prefill_size=1,
            )


if __name__ == "__main__":
    unittest.main()
