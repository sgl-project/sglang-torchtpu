# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the GLM-5.3-Flash KV state the direct tracer lifts.

The direct tracer reads the target's caches through the pool's get_state() and
swaps in placeholders with set_state(); a buffer the forward reads outside that
list enters the trace as an unprovided tensor. GLM-5.3-Flash keeps its DSA
layers in TpuSplitMLATokenToKVPool and its KDA state in TpuMambaPool, lifted
together by hybrid_linear_kv_pool_state. Guards that the state covers every
cache accessor, stays in one fixed order, and that set_state rebinds what the
accessors return.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.configs.mamba_utils import KimiLinearCacheParams, KimiLinearStateShape
from sglang.srt.mem_cache.kv.state_storage import StateStorage
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.compilation.hybrid_linear_direct_compile import (
    hybrid_linear_kv_pool_state,
)
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.kv_pool import TpuSplitMLATokenToKVPool
from sglang_tpu.mamba_pool import TpuMambaPool
from sglang_tpu.models.glm5_next_compile_policy import Glm5NextDecoderTpuCompilePolicy

PAGE_SIZE = 16
NUM_TOKENS = 64
# A layer that reuses the previous layer's top-k has no index-key or raw cache.
INDEX_KEY_LAYERS = [True, False, True]
KDA_LAYERS = [0, 1, 2, 4]


def _split_pool() -> TpuSplitMLATokenToKVPool:
    return TpuSplitMLATokenToKVPool(
        size=NUM_TOKENS,
        page_size=PAGE_SIZE,
        dtype=torch.float8_e4m3fn,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=len(INDEX_KEY_LAYERS),
        device="cpu",
        enable_memory_saver=False,
        index_head_dim=128,
        index_key_layers=INDEX_KEY_LAYERS,
        index_kpool=4,
    )


def _accessor_tensors(pool: TpuSplitMLATokenToKVPool) -> list[torch.Tensor]:
    """Every tensor the sparse MLA ops and the DSA indexer read through the pool."""
    tensors = []
    for layer, has_indexer in enumerate(INDEX_KEY_LAYERS):
        tensors += [pool.get_nope_cache(layer), pool.get_rope_cache(layer)]
        if has_indexer:
            tensors += [pool.get_index_key_cache(layer)]
            tensors += [pool.get_kpool_raw_cache(layer)]
    return tensors


def _mamba_pool() -> TpuMambaPool:
    shape = KimiLinearStateShape.create(
        tp_world_size=1, num_heads=2, head_dim=8, conv_kernel_size=4
    )
    return TpuMambaPool(
        size=4,
        spec_state_size=4,
        cache_params=KimiLinearCacheParams(shape=shape, layers=KDA_LAYERS),
        mamba_layer_ids=KDA_LAYERS,
        device="cpu",
    )


class TestSplitMLAPoolState(CustomTestCase):
    def setUp(self):
        self.pool = _split_pool()

    def test_state_covers_every_accessor_tensor(self):
        state_ids = {id(t) for t in self.pool.get_state()}
        missing = [
            tuple(t.shape)
            for t in _accessor_tensors(self.pool)
            if id(t) not in state_ids
        ]
        self.assertEqual(missing, [])
        self.assertEqual(len(state_ids), len(_accessor_tensors(self.pool)))

    def test_state_is_base_tensors_in_a_fixed_order(self):
        first = self.pool.get_state()
        self.assertTrue(all(t._base is None for t in first))
        self.assertEqual(len({id(t) for t in first}), len(first))
        self.assertEqual([id(t) for t in first], [id(t) for t in self.pool.get_state()])

    def test_set_state_rebinds_the_accessors(self):
        replacement = [torch.empty_like(t) for t in self.pool.get_state()]
        self.pool.set_state(replacement)
        self.assertEqual(
            [id(t) for t in self.pool.get_state()], [id(t) for t in replacement]
        )
        replacement_ids = {id(t) for t in replacement}
        self.assertTrue(
            all(id(t) in replacement_ids for t in _accessor_tensors(self.pool))
        )
        # A layer without an indexer keeps no index-key or raw cache.
        self.assertIsNone(self.pool.index_key_buffer[1])
        self.assertIsNone(self.pool.kpool_raw_buffer[1])

    def test_set_state_rejects_a_wrong_length(self):
        state = self.pool.get_state()
        for wrong in (state[:-1], [*state, state[0]]):
            with self.assertRaises(ValueError):
                self.pool.set_state(wrong)

    def test_attached_raw_ring_is_state(self):
        """With SGLANG_TPU_DSA_KPOOL_RAW_RING the raw rows live in a per-slot
        ring the attention backend attaches before the first compile."""
        with (
            tpu_envs.SGLANG_TPU_DSA_KPOOL_RAW_RING.override(True),
            get_context().override_server_args(chunked_prefill_size=32),
        ):
            pool = _split_pool()
        self.assertGreater(pool.raw_ring_pages_cnt, 0)
        before = len(pool.get_state())
        pool.attach_raw_ring(5)
        state_ids = {id(t) for t in pool.get_state()}
        self.assertEqual(len(state_ids), before + sum(INDEX_KEY_LAYERS))
        self.assertTrue(all(id(t) in state_ids for t in _accessor_tensors(pool)))


class TestHybridSplitPoolState(CustomTestCase):
    def test_split_mla_state_then_kda_state(self):
        """A split at the wrong index still traces, but hands MLA pages to the
        KDA layers (or the reverse) and silently corrupts both caches."""
        split = _split_pool()
        mamba = _mamba_pool()
        kv_pool = object.__new__(HybridLinearKVPool)
        kv_pool.full_kv_pool = split
        req_pool = object.__new__(HybridReqToTokenPool)
        # mamba_pool is a view of the request table's state storage.
        req_pool.state = StateStorage.empty(req_pool.state_sub_pool)
        req_pool.mamba_pool = mamba
        state = hybrid_linear_kv_pool_state(kv_pool=kv_pool, req_to_token_pool=req_pool)

        expected = [*split.get_state(), *mamba.get_state()]
        self.assertEqual([id(t) for t in state.get_state()], [id(t) for t in expected])
        replacement = [torch.empty_like(t) for t in expected]
        state.set_state(replacement)
        num_split = len(split.get_state())
        self.assertEqual(
            [id(t) for t in split.get_state()],
            [id(t) for t in replacement[:num_split]],
        )
        self.assertEqual(
            [id(t) for t in mamba.get_state()],
            [id(t) for t in replacement[num_split:]],
        )


class TestGlm5NextDecoderPolicy(CustomTestCase):
    def test_declares_spmd_safe_over_the_fallback(self):
        self.assertTrue(
            issubclass(Glm5NextDecoderTpuCompilePolicy, DecoderTpuCompilePolicy)
        )
        self.assertTrue(Glm5NextDecoderTpuCompilePolicy.spmd_safe)
        self.assertFalse(DecoderTpuCompilePolicy.spmd_safe)


if __name__ == "__main__":
    unittest.main()
