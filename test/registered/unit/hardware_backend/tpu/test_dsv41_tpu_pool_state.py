# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the DeepSeek-V4.1 TPU KV pool's direct_compile state.

The direct tracer lifts the pool through get_state() and swaps in placeholders
with set_state(); any buffer the forward reads outside that list enters the trace
as an unregistered device tensor. Guards that every kernel accessor reads from the
state, and that set_state() rebinds what the accessors return.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import unittest

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41.memory_pool import DSV41TPUTokenToKVPool

RATIOS = [0, 0] + [2] * 18 + [1] * 20
SOURCES = [2, 8, 14, 20]


def _make_pool(testcase: unittest.TestCase) -> DSV41TPUTokenToKVPool:
    from sglang.srt.runtime_context import get_context

    override = get_context().override_server_args(page_size=256)
    override.install()
    testcase.addCleanup(override.restore)
    return DSV41TPUTokenToKVPool(
        max_num_reqs=4,
        num_req_slots=5,
        swa_size=1024,
        c4_size=0,
        c128_size=0,
        c4_state_pool_size=0,
        c128_state_pool_size=0,
        page_size=256,
        swa_page_size=256,
        sliding_window=128,
        dtype=torch.bfloat16,
        c4_state_dtype=torch.float32,
        c128_state_dtype=torch.float32,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        indexer_head_dim=128,
        layer_num=len(RATIOS),
        device="cpu",
        enable_memory_saver=False,
        compression_ratios=RATIOS,
        start_layer=0,
        end_layer=len(RATIOS),
        kv_source_layers=SOURCES,
        full_size=4096,
    )


def _accessor_tensors(pool: DSV41TPUTokenToKVPool) -> list[torch.Tensor]:
    """Every tensor the V4.1 attention backend reads through the pool."""
    tensors = [pool.get_swa_pages(layer) for layer in range(len(RATIOS))]
    for source in SOURCES:
        tensors.extend(pool.get_main_kv(source))
        tensors.append(pool.get_index_k(source))
    # The ratio-2 pending-pair ring lives on each ratio-2 kv_source layer.
    for source in SOURCES:
        if RATIOS[source] == 2:
            state = pool.get_attention_compress_states(source)
            tensors.append(state.kv_score_buffer.kv_score)
    return tensors


class TestDSV41TPUPoolState(CustomTestCase):
    def setUp(self):
        self.pool = _make_pool(self)

    def test_state_covers_every_accessor_tensor(self):
        state_ids = {id(t) for t in self.pool.get_state()}
        missing = [
            t.shape for t in _accessor_tensors(self.pool) if id(t) not in state_ids
        ]
        self.assertEqual(missing, [])

    def test_state_is_base_tensors_in_a_fixed_order(self):
        first = self.pool.get_state()
        self.assertTrue(all(t._base is None for t in first))
        self.assertEqual(len({id(t) for t in first}), len(first))
        self.assertEqual([id(t) for t in first], [id(t) for t in self.pool.get_state()])

    def test_set_state_rebinds_the_accessors(self):
        replacement = [torch.empty_like(t) for t in self.pool.get_state()]
        self.pool.set_state(replacement)
        replacement_ids = {id(t) for t in replacement}
        self.assertEqual(
            [id(t) for t in self.pool.get_state()], [id(t) for t in replacement]
        )
        self.assertTrue(
            all(id(t) in replacement_ids for t in _accessor_tensors(self.pool))
        )

    def test_set_state_rejects_a_wrong_length(self):
        with self.assertRaises(ValueError):
            self.pool.set_state(self.pool.get_state()[:-1])


if __name__ == "__main__":
    unittest.main()
