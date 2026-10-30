# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Logical-to-Physical TPU Unified SWA/Full KV cache pool."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool, TpuSeqAlongLaneKVPool
from sglang_tpu.unified_kv_pool import (
    AttnLayerSpec,
    TpuUnifiedSWAKVPool,
    UnifiedPageLayout,
    num_swa_groups,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTpuUnifiedSWAKVPoolStorage(unittest.TestCase):
    """Tests for PR 1: shared physical buffers and SWA group page mapping."""

    def test_equal_full_and_swa_layers_share_buffers(self):
        pool = TpuUnifiedSWAKVPool(
            size=64,
            page_size=4,
            dtype=torch.bfloat16,
            full=AttnLayerSpec(head_num=4, head_dim=128, layer_ids=[1, 3]),
            swa=AttnLayerSpec(head_num=4, head_dim=128, layer_ids=[0, 2]),
            device="cpu",
        )
        self.assertIsInstance(pool, BaseSWAKVPool)
        self.assertNotIsInstance(pool, SWAKVPool)
        self.assertEqual(pool.size, 64)
        self.assertEqual(
            pool.layout,
            UnifiedPageLayout(
                page_head_num=4,
                page_head_dim=128,
                num_shared_buffers=2,
                num_swa_groups=1,
            ),
        )
        self.assertEqual(len(pool.get_state()), 2)
        for full_lid, swa_lid in ((1, 0), (3, 2)):
            self.assertEqual(
                pool.get_fused_kv_buffer(full_lid).data_ptr(),
                pool.get_fused_kv_buffer(swa_lid).data_ptr(),
            )
            self.assertEqual(pool.swa_page_group(swa_lid), 0)
        self.assertEqual(pool.get_kv_size_bytes(), pool.kv_pool.get_kv_size_bytes())
        self.assertEqual(pool.get_state_buf_infos(), ([], [], []))

    def test_multi_group_swa_ratio_bf16_and_fp8(self):
        full_ids = [5, 11]
        swa_ids = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                pool = TpuUnifiedSWAKVPool(
                    size=80,
                    page_size=4,
                    dtype=dtype,
                    full=AttnLayerSpec(head_num=2, head_dim=256, layer_ids=full_ids),
                    swa=AttnLayerSpec(head_num=2, head_dim=256, layer_ids=swa_ids),
                    device="cpu",
                )
                self.assertEqual(
                    (pool.layout.num_swa_groups, pool.layout.num_shared_buffers),
                    (5, 2),
                )
                self.assertEqual(pool.kv_pool.num_blocks, 21)
                self.assertEqual(pool.swa_group_pages.shape, (4, 21))
                for j, lid in enumerate(swa_ids):
                    self.assertEqual(pool.swa_page_group(lid), j // 2)
                    self.assertEqual(
                        pool.get_fused_kv_buffer(lid).data_ptr(),
                        pool.get_state()[j % 2].data_ptr(),
                    )
                new_state = [torch.zeros_like(buf) for buf in pool.get_state()]
                pool.set_state(new_state)
                for j, lid in enumerate(swa_ids):
                    self.assertEqual(
                        pool.get_fused_kv_buffer(lid).data_ptr(),
                        new_state[j % 2].data_ptr(),
                    )

    def test_mismatched_page_shapes_rejected(self):
        for swa_heads, swa_dim in ((1, 256), (4, 256), (2, 128), (2, 512)):
            with self.subTest(swa_heads=swa_heads, swa_dim=swa_dim):
                with self.assertRaises(ValueError):
                    TpuUnifiedSWAKVPool(
                        size=32,
                        page_size=4,
                        dtype=torch.bfloat16,
                        full=AttnLayerSpec(head_num=2, head_dim=256, layer_ids=[2]),
                        swa=AttnLayerSpec(
                            head_num=swa_heads, head_dim=swa_dim, layer_ids=[0, 1]
                        ),
                        device="cpu",
                    )

    def test_non_divisible_swa_groups_and_cpu_copy_restore(self):
        self.assertEqual(num_swa_groups(8, 20), 3)
        page_size = 4
        pool = TpuUnifiedSWAKVPool(
            size=48,
            page_size=page_size,
            dtype=torch.bfloat16,
            full=AttnLayerSpec(head_num=2, head_dim=128, layer_ids=range(8)),
            swa=AttnLayerSpec(head_num=2, head_dim=128, layer_ids=range(8, 28)),
            device="cpu",
        )
        self.assertEqual(
            (pool.layout.num_shared_buffers, pool.layout.num_swa_groups), (8, 3)
        )
        # Prefix Full page 1 (unmapped in SWA) + tail Full page 2 -> SWA primary page 3
        # with companion pages [9, 11], restored into Full pages [4, 5] and SWA [6, 2, 10].
        mapping = torch.zeros(48 + page_size + 1, dtype=torch.int64)
        mapping[-1] = -1
        mapping[2 * page_size : 3 * page_size] = torch.arange(
            3 * page_size, 4 * page_size
        )
        pool.register_mapping(mapping)
        self.assertTrue(
            torch.equal(
                pool.translate_loc_from_full_to_swa(torch.tensor([8, 9, -1])),
                torch.tensor([12, 13, -1], dtype=torch.int32),
            )
        )
        pool.swa_group_pages[:, 3] = torch.tensor([9, 11], dtype=torch.int32)
        for i in range(8):
            pool.get_state()[i][1].fill_(float(50 + i))
            pool.get_state()[i][2].fill_(float(100 + i))
        for j in range(20):
            p = 3 if (j // 8) == 0 else int(pool.swa_group_pages[j // 8 - 1, 3])
            pool.get_state()[j % 8][p].fill_(float(10 + j))

        src_full = torch.arange(1 * page_size, 3 * page_size)
        cpu_copy = pool.get_cpu_copy(src_full)

        mapping.zero_()
        mapping[-1] = -1
        mapping[5 * page_size : 6 * page_size] = torch.arange(
            6 * page_size, 7 * page_size
        )
        pool.swa_group_pages[:, 6] = torch.tensor([2, 10], dtype=torch.int32)
        dst_full = torch.arange(4 * page_size, 6 * page_size)
        pool.load_cpu_copy(cpu_copy, dst_full)

        for i in range(8):
            self.assertTrue(torch.all(pool.get_state()[i][4] == float(50 + i)))
            self.assertTrue(torch.all(pool.get_state()[i][5] == float(100 + i)))
        for j in range(20):
            p = 6 if (j // 8) == 0 else int(pool.swa_group_pages[j // 8 - 1, 6])
            self.assertTrue(torch.all(pool.get_state()[j % 8][p] == float(10 + j)))

        # Verify Raiden retraction backup stages Full pages via handle while SWA uses row-copy
        from sglang_tpu.raiden_kv_backup import RaidenKVBackupHandle

        saved_indices = []
        restored = []

        class _FakeRaidenBackup:
            def _free_pages(self, pages):
                pass

            def try_save(self, idx):
                saved_indices.append(idx.tolist())
                return RaidenKVBackupHandle(self, [0, 1], 0, idx.numel())

            def restore(self, handle, idx):
                restored.append((list(handle.host_pages), idx.tolist()))

        pool.kv_pool._raiden_backup_obj = _FakeRaidenBackup()
        raiden_copy = pool.get_cpu_copy(dst_full)
        self.assertIsInstance(raiden_copy["full"], RaidenKVBackupHandle)
        self.assertEqual(len(saved_indices), 1)
        self.assertIsInstance(raiden_copy["swa"], list)
        # Shrink SWA mapping on restore to exercise filter_cpu_copy alongside Raiden Full restore
        mapping[5 * page_size] = 0
        pool.load_cpu_copy(raiden_copy, dst_full)
        self.assertEqual(restored, [([0, 1], dst_full.tolist())])

        # Verify TpuSeqAlongLaneKVPool saves 50% HBM vs TpuFusedMHATokenToKVPool at head_num=1 in fp8
        with patch(
            "sglang_tpu.kv_pool._seq_along_lane_head_dim",
            side_effect=lambda *, head_dim, kv_packing: (
                ((head_dim + 8 * kv_packing - 1) // (8 * kv_packing)) * (8 * kv_packing)
            ),
        ):
            pool_across = TpuUnifiedSWAKVPool(
                size=256,
                page_size=128,
                dtype=torch.float8_e4m3fn,
                full=AttnLayerSpec(head_num=1, head_dim=128, layer_ids=[0]),
                swa=AttnLayerSpec(head_num=1, head_dim=128, layer_ids=[1, 2]),
                device="cpu",
                token_to_kv_pool_class=TpuFusedMHATokenToKVPool,
            )
            pool_along = TpuUnifiedSWAKVPool(
                size=256,
                page_size=128,
                dtype=torch.float8_e4m3fn,
                full=AttnLayerSpec(head_num=1, head_dim=128, layer_ids=[0]),
                swa=AttnLayerSpec(head_num=1, head_dim=128, layer_ids=[1, 2]),
                device="cpu",
                token_to_kv_pool_class=TpuSeqAlongLaneKVPool,
            )
        self.assertEqual(
            pool_along.kv_pool.kv_buffer[0].numel() * 2,
            pool_across.kv_pool.kv_buffer[0].numel(),
        )


if __name__ == "__main__":
    unittest.main()
