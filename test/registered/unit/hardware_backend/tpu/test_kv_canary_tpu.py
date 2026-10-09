"""kv-canary attachers for the TPU fused KV pools (#1176)."""

import unittest

import torch

from sglang.srt.kv_canary.buffer_group import PoolKind
from sglang.srt.kv_canary.pool_patcher import api
from sglang.srt.kv_canary.pool_patcher.adapters import swa
from sglang.srt.kv_canary.pool_patcher.adapters.swa import (
    attach_swa,
    register_subpool_group_builder,
)
from sglang.srt.platforms.interface import SRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.kv_canary_tpu import (
    attach_tpu_fused,
    build_fused_group,
    register_tpu_attachers,
)
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool, TpuSeqAlongLaneKVPool
from sglang_tpu.tpu_platform import TpuSRTPlatform

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FusedPool:
    """Stand-in for TpuFusedMHATokenToKVPool: one fused buffer per layer, no k/v lists."""

    def __init__(self, *, num_blocks=4, page_size=8, layers=2):
        self.page_size = page_size
        # [num_blocks, page_size, heads_x2 // packing, packing, head_dim] -> 128 B per slot
        self.kv_buffer = [
            torch.zeros(num_blocks, page_size, 2, 2, 16, dtype=torch.bfloat16)
            for _ in range(layers)
        ]

    def get_contiguous_buf_infos(self):
        return (
            [b.data_ptr() for b in self.kv_buffer],
            [b.nbytes for b in self.kv_buffer],
            [b[0].nbytes for b in self.kv_buffer],
        )


class _FusedSwaPool:
    """Stand-in for SWAKVPool whose sub-pools are TPU fused pools."""

    def __init__(self):
        self.page_size = 8
        self.full_kv_pool = _FusedPool(num_blocks=4, layers=2)
        self.swa_kv_pool = _FusedPool(num_blocks=2, layers=3)
        self.full_to_swa_index_mapping = torch.zeros(33, dtype=torch.int64)

    def get_contiguous_buf_infos(self):
        return self.full_kv_pool.get_contiguous_buf_infos()

    def get_state_buf_infos(self):
        return self.swa_kv_pool.get_contiguous_buf_infos()


class TestCanaryTpuFusedPoolAttacher(CustomTestCase):
    """#1176: --kv-canary could not attach to the TPU fused KV pools."""

    def test_platform_registration_covers_both_tpu_pool_classes(self):
        # The TPU platform runs this from init_backend(); the shared patcher
        # and SWA adapter then find the fused pools in their registries.

        register_tpu_attachers()
        for cls in (TpuFusedMHATokenToKVPool, TpuSeqAlongLaneKVPool):
            self.assertIs(api._POOL_ATTACHERS[cls], attach_tpu_fused)
            self.assertIs(swa._SUBPOOL_GROUP_BUILDERS[cls], build_fused_group)

    def test_fused_pool_gets_one_canary_pair_per_token_slot(self):
        pool = _FusedPool(num_blocks=4, page_size=8, layers=2)
        (group,) = attach_tpu_fused(
            pool=pool,
            device=torch.device("cpu"),
            read_bytes=16,
            kv_token_id_vs_position_offset=0,
        )
        self.assertIs(group.kind, PoolKind.FULL)
        self.assertEqual(group.k_head.shape[0], 4 * 8)  # num_blocks * page_size slots
        self.assertIsNone(group.v_head)
        self.assertFalse(group.has_v_half)
        self.assertEqual(len(group.real_kv_sources_k), 1)
        self.assertEqual(group.real_kv_sources_k[0].num_bytes_per_token, 128)
        self.assertEqual(group.real_kv_sources_v, ())
        # The transfer buf infos gain the head and tail canary buffers around the layers.
        ptrs, lens, item_lens = pool.get_contiguous_buf_infos()
        self.assertEqual(len(ptrs), 2 + 2)
        self.assertEqual(
            (ptrs[0], ptrs[-1]), (group.k_head.data_ptr(), group.k_tail.data_ptr())
        )
        self.assertEqual(item_lens[0], group.k_head[0].nbytes * pool.page_size)

    def test_swa_adapter_handles_fused_sub_pools(self):
        # The fake stands in for the TPU pool class the platform registers.
        register_subpool_group_builder(_FusedPool, build_fused_group)
        pool = _FusedSwaPool()
        full, swa = attach_swa(
            pool=pool,
            device=torch.device("cpu"),
            read_bytes=16,
            kv_token_id_vs_position_offset=0,
        )
        self.assertEqual((full.kind, swa.kind), (PoolKind.FULL, PoolKind.SWA))
        self.assertEqual((full.k_head.shape[0], swa.k_head.shape[0]), (32, 16))
        self.assertIs(swa.swa_index_lut, pool.full_to_swa_index_mapping)
        self.assertFalse(full.has_v_half)
        # Odd layer count on the SWA sub-pool: a K/V split would have been refused.
        ptrs, _, _ = pool.get_state_buf_infos()
        self.assertEqual(len(ptrs), 3 + 2)

    def test_tpu_platform_declines_the_canary_runtime(self):
        self.assertFalse(
            TpuSRTPlatform.supports_kv_canary(TpuSRTPlatform.__new__(TpuSRTPlatform))
        )
        self.assertTrue(
            SRTPlatform.supports_kv_canary(SRTPlatform.__new__(SRTPlatform))
        )


if __name__ == "__main__":
    unittest.main()
