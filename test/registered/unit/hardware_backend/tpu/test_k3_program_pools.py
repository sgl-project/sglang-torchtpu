"""K3_PROGRAM's sglang pool classes over the unified pools: built in sglang's
order (state pool, then KV pool), sized from sglang's counts, and described to
the P/D transfer through HybridLinearKVPool."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.raiden.common import get_raiden_kv_caches
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.hooks import k3_program as hooks
from sglang_tpu.k3_programs import sglang_bridge as bridge
from sglang_tpu.k3_programs import sglang_pools
from sglang_tpu.k3_programs.kimi_k3 import pools
from sglang_tpu.k3_programs.kimi_k3.layout import KDA, MLA, NUM_LAYERS

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

STEP = 8192
BLOCK = 128 * 2 * 640 * 2  # bf16 bytes


def _model(kv_cache_dtype="auto"):
    return mock.patch(
        "sglang.srt.runtime_context.get_model",
        return_value=SimpleNamespace(kv_cache_dtype=kv_cache_dtype),
    )


def _build(slots=4, pages=10, page_size=STEP):
    state = sglang_pools.K3StatePool(size=slots, mamba_layer_ids=KDA, device="cpu")
    kv = sglang_pools.K3UnifiedKVPool(
        size=pages * page_size,
        page_size=page_size,
        dtype=torch.bfloat16,
        layer_num=len(MLA),
        device="cpu",
    )
    return state, kv


class TestUnifiedPoolClasses(CustomTestCase):
    def test_sizes_follow_sglang_counts(self):
        with _model():
            state, kv = _build(slots=4, pages=10)
        self.assertEqual((kv.plan.state_slots, kv.plan.kv_blocks), (4, 10))
        self.assertEqual(kv.size, 10 * STEP)
        self.assertIs(state.plan, kv.plan)
        self.assertEqual(len(kv.raw_pools()), len(MLA) + 5)
        self.assertEqual(
            kv.get_kv_size_bytes(), (len(MLA) + 5) * kv.plan.num_blocks * BLOCK
        )
        with _model():
            self.assertEqual(
                sglang_pools.K3StatePool.cache_bytes_per_request(None),
                3 * (len(MLA) + 5) * BLOCK,
            )
        # The program binds the pools the KV pool allocated.
        raw, *_ = pools.bind_raw_pools(kv.caches)
        self.assertEqual(len(raw), len(MLA))
        self.assertEqual(len(pools.draft_raw_pools(kv.caches)), 5)

    def test_fp8_pages_hold_twice_the_tokens(self):
        with _model("fp8_e4m3"):
            _, kv = _build(page_size=2 * STEP)
        self.assertEqual(kv.plan.logical_block_tokens, 2 * STEP)
        with _model("fp8_e4m3"), self.assertRaisesRegex(ValueError, "16384 tokens"):
            _build(page_size=STEP)

    def test_order_and_shape_checks(self):
        sglang_pools._pending_state_pool = None
        with _model(), self.assertRaisesRegex(RuntimeError, "state pool first"):
            sglang_pools.K3UnifiedKVPool(
                size=STEP,
                page_size=STEP,
                dtype=torch.bfloat16,
                layer_num=24,
                device="cpu",
            )
        with self.assertRaisesRegex(ValueError, "not Kimi-K3's KDA layers"):
            sglang_pools.K3StatePool(size=4, mamba_layer_ids=[0, 1], device="cpu")

    def test_transfer_descriptions_through_the_hybrid_pool(self):
        with _model():
            state, kv = _build()
        hybrid = HybridLinearKVPool(
            size=kv.size,
            dtype=torch.bfloat16,
            page_size=STEP,
            head_num=1,
            head_dim=576,
            full_attention_layer_ids=list(MLA),
            device="cpu",
            mamba_pool=state,
            use_mla=True,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            full_kv_pool=kv,
        )
        kv_ptrs, _, kv_items = hybrid.get_contiguous_buf_infos()
        self.assertEqual(len(kv_ptrs), len(MLA) + 5)
        self.assertEqual(set(kv_items), {BLOCK})
        state_ptrs, _, state_items = hybrid.get_state_buf_infos()
        self.assertEqual(len(state_ptrs), len(KDA))
        self.assertEqual(set(state_items), {3 * BLOCK})
        self.assertEqual(hybrid.get_state_layer_ids(), list(KDA))
        # Core's ids cover the 24 MLA entries only; the hook covers all 29.
        ids = hooks.k3_program_kv_layer_ids(HybridLinearKVPool.get_kv_layer_ids, hybrid)
        self.assertEqual(len(ids), len(kv_ptrs))
        self.assertEqual(ids[-5:], list(range(NUM_LAYERS, NUM_LAYERS + 5)))
        self.assertEqual(ids[: len(MLA)], list(MLA))


class TestRaidenGroups(CustomTestCase):
    """Raiden moves rows of root tensors by index: a page's row is its page
    id, a slot's three rows are the blocks the program reads its state from."""

    def setUp(self):
        with _model():
            self.state, self.kv = _build(slots=4, pages=10)
        self.hybrid = HybridLinearKVPool(
            size=self.kv.size,
            dtype=torch.bfloat16,
            page_size=STEP,
            head_num=1,
            head_dim=576,
            full_attention_layer_ids=list(MLA),
            device="cpu",
            mamba_pool=self.state,
            use_mla=True,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            full_kv_pool=self.kv,
        )

    def test_groups(self):
        groups = get_raiden_kv_caches(self.hybrid)
        plan = self.kv.plan
        self.assertEqual(len(groups.full), len(MLA) + 5)
        self.assertEqual(groups.num_draft_full, 5)
        self.assertEqual(
            groups.full_layer_ids[-5:], list(range(NUM_LAYERS, NUM_LAYERS + 5))
        )
        self.assertTrue(all(not t._is_view() for t in groups.full + groups.state))
        self.assertEqual(len(groups.state), len(KDA) // 3)
        self.assertEqual(
            {int(t.shape[0]) for t in groups.full + groups.state}, {plan.num_blocks}
        )
        self.assertEqual(groups.state_max_blocks, 3)
        # The draft pools come after the target's, as Raiden's draft entries do.
        self.assertEqual(
            [t.data_ptr() for t in groups.full[-5:]],
            [t.data_ptr() for t in pools.draft_raw_pools(self.kv.caches)],
        )

    def test_slot_rows_are_the_programs_state_blocks(self):
        groups = get_raiden_kv_caches(self.hybrid)
        manager = RaidenKVManager.__new__(RaidenKVManager)
        manager._state_engine = object()
        manager._state_type = StateType.MAMBA
        manager.kv_args = SimpleNamespace(state_types=[StateType.MAMBA])
        manager._committed_state_row = groups.committed_state_row
        manager._state_engine_max_blocks = groups.state_max_blocks
        plan = self.kv.plan
        for slot in (1, 4):
            rows = manager._state_rows_from_state_indices([[slot]])
            self.assertEqual(rows, [plan.state_block(slot, g) for g in range(3)])
            # KDA layer k reads its state from row state_block(slot, k % 3) of
            # pool k // 3: one of the rows moved, in the pool registered for it.
            tables = bridge.block_tables(
                np.arange(STEP, 2 * STEP), STEP, slot, plan, table_width=4
            )
            groups_kv = bridge.kv_groups(plan, 1 << 20, 4)
            of = {n: g for g, grp in enumerate(groups_kv) for n in grp.layer_names}
            kda = [(of[bridge.kda_layer_name(layer)], 1 << 20) for layer in KDA]
            blocks = pools.kda_blocks_at_length(tables, kda, 5)
            for k, block in enumerate(blocks.tolist()):
                self.assertIn(block, rows)
                self.assertEqual(
                    groups.state[k // 3].data_ptr(),
                    self.kv.caches[bridge.kda_layer_name(KDA[k])].data_ptr(),
                )


class TestSpecCheckHook(CustomTestCase):
    def test_only_k3_program_skips_the_check(self):
        calls = []

        def original(server_args):
            calls.append(server_args)

        k3 = SimpleNamespace(speculative_algorithm="K3_PROGRAM")
        dspark = SimpleNamespace(speculative_algorithm="DSPARK")
        with mock.patch(
            "sglang.srt.arg_groups.overrides.resolving_view", side_effect=lambda a: a
        ):
            hooks.allow_k3_program_spec(original, k3)
            hooks.allow_k3_program_spec(original, dspark)
        self.assertEqual(calls, [dspark])


if __name__ == "__main__":
    unittest.main()
