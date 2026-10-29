"""K3_PROGRAM's sglang pool classes over the unified pools: built in sglang's
order (state pool, then KV pool), sized from sglang's counts, and described to
the P/D transfer through HybridLinearKVPool."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.hooks import k3_program as hooks
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
