"""CPU unit tests for Raiden P/D over the TPU split sparse-MLA pool.

GLM-5.3-Flash keeps its DSA layers in ``TpuSplitMLATokenToKVPool`` (several
page buffers per layer) under a ``HybridLinearKVPool`` whose KDA state lives in
``TpuMambaPool``; its MTP draft gets its own one-layer split pool on the
target's page ids. These tests pin what P/D registers for that layout.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.configs.mamba_utils import KimiLinearCacheParams, KimiLinearStateShape
from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.utils import build_kv_layer_ids, setup_state_kv_args
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.raiden.common import get_raiden_kv_caches
from sglang_tpu.kv_pool import TpuSplitMLATokenToKVPool
from sglang_tpu.mamba_pool import TpuMambaPool

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 16
NUM_TOKENS = 64
# Global ids of the target's two DSA layers; the KDA layers are the others.
DSA_LAYERS = [3, 7]
KDA_LAYERS = [0, 1, 2, 4, 5, 6]
NUM_HIDDEN_LAYERS = 8


def _split_pool(layer_num: int, qk_rope_head_dim: int = 0) -> TpuSplitMLATokenToKVPool:
    return TpuSplitMLATokenToKVPool(
        size=NUM_TOKENS,
        page_size=PAGE_SIZE,
        dtype=torch.float8_e4m3fn,
        kv_lora_rank=512,
        qk_rope_head_dim=qk_rope_head_dim,
        layer_num=layer_num,
        device="cpu",
        enable_memory_saver=False,
        index_head_dim=128,
        index_key_layers=[True] * layer_num,
        index_kpool=4,
    )


def _hybrid_pool(target: TpuSplitMLATokenToKVPool) -> HybridLinearKVPool:
    shape = KimiLinearStateShape.create(
        tp_world_size=1, num_heads=2, head_dim=8, conv_kernel_size=4
    )
    mamba = TpuMambaPool(
        size=4,
        spec_state_size=4,
        cache_params=KimiLinearCacheParams(shape=shape, layers=KDA_LAYERS),
        mamba_layer_ids=KDA_LAYERS,
        device="cpu",
    )
    # Only the attributes the P/D registration paths read; the real
    # constructor needs a model config.
    pool = object.__new__(HybridLinearKVPool)
    pool.full_kv_pool = target
    pool.mamba_pool = mamba
    pool.full_attention_layer_id_mapping = {
        layer: dense for dense, layer in enumerate(DSA_LAYERS)
    }
    pool.use_dsa = True
    return pool


def _kind_major(pool: TpuSplitMLATokenToKVPool, *kinds: str) -> list[torch.Tensor]:
    return [buf for kind in kinds for buf in getattr(pool, kind)]


class TestSplitMLATransferBuffers(CustomTestCase):
    def test_nope_model_leaves_rope_behind(self):
        pool = _split_pool(layer_num=2)
        entries = pool.transfer_page_buffers()
        self.assertEqual([layer for layer, _ in entries], [0, 1] * 3)
        expected = _kind_major(
            pool, "nope_buffer", "index_key_buffer", "kpool_raw_buffer"
        )
        self.assertEqual(len(entries), len(expected))
        for (_, buf), want in zip(entries, expected):
            self.assertIs(buf, want)

        ptrs, lens, item_lens = pool.get_contiguous_buf_infos()
        self.assertEqual(ptrs, [buf.data_ptr() for buf in expected])
        self.assertEqual(lens, [buf.nbytes for buf in expected])
        # One item is one page: NoPE 512 B, index key 64 B, raw key|gate 512 B
        # per token.
        per_token = [512, 512, 64, 64, 512, 512]
        self.assertEqual(item_lens, [PAGE_SIZE * b for b in per_token])

    def test_rope_model_keeps_rope(self):
        pool = _split_pool(layer_num=2, qk_rope_head_dim=64)
        expected = _kind_major(
            pool, "nope_buffer", "rope_buffer", "index_key_buffer", "kpool_raw_buffer"
        )
        bufs = [buf for _, buf in pool.transfer_page_buffers()]
        self.assertEqual(len(bufs), len(expected))
        for buf, want in zip(bufs, expected):
            self.assertIs(buf, want)


class TestHybridSplitMLARaidenPD(CustomTestCase):
    def setUp(self):
        self.target = _split_pool(layer_num=len(DSA_LAYERS))
        self.draft = _split_pool(layer_num=1)
        self.pool = _hybrid_pool(self.target)

    def test_raiden_groups_target_draft_and_kda_state(self):
        groups = get_raiden_kv_caches(self.pool, self.draft)

        target_bufs = [buf for _, buf in self.target.transfer_page_buffers()]
        draft_bufs = [buf for _, buf in self.draft.transfer_page_buffers()]
        self.assertEqual(len(groups.full), len(target_bufs) + len(draft_bufs))
        for buf, want in zip(groups.full, target_bufs + draft_bufs):
            self.assertIs(buf, want)
        self.assertEqual(groups.swa, [])
        self.assertEqual(groups.num_draft_full, len(draft_bufs))
        self.assertEqual(groups.full_layer_ids, DSA_LAYERS * 3 + [0] * 3)

        self.assertEqual(groups.state_type, StateType.MAMBA)
        self.assertEqual(len(groups.state), 2 * len(KDA_LAYERS))
        for buf, want in zip(groups.state, self.pool.mamba_pool.get_state()):
            self.assertIs(buf, want)
        self.assertEqual(groups.committed_state_row(3), 3)

    def test_state_kv_args_carry_only_kda_state(self):
        kv_args = KVArgs()
        setup_state_kv_args(kv_args, self.pool, self.draft)

        self.assertEqual(kv_args.state_types, [StateType.MAMBA])
        ptrs, _, _ = self.pool.mamba_pool.get_contiguous_buf_infos()
        self.assertEqual(kv_args.state_data_ptrs, [ptrs])
        self.assertTrue(kv_args.is_hybrid_mla_backend)

    def test_kv_layer_ids_cover_target_and_draft_entries(self):
        target_ptrs, _, _ = self.pool.get_contiguous_buf_infos()
        draft_ptrs, _, _ = self.draft.get_contiguous_buf_infos()

        layer_ids = build_kv_layer_ids(
            token_to_kv_pool=self.pool,
            draft_token_to_kv_pool=self.draft,
            num_draft_entries=len(draft_ptrs),
            num_hidden_layers=NUM_HIDDEN_LAYERS,
        )

        self.assertEqual(len(layer_ids), len(target_ptrs) + len(draft_ptrs))
        self.assertEqual(
            layer_ids, DSA_LAYERS * 3 + [NUM_HIDDEN_LAYERS] * len(draft_ptrs)
        )


if __name__ == "__main__":
    unittest.main()
