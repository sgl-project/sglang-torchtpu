# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Logical-to-Physical TPU Unified SWA/Full KV cache pool."""

from __future__ import annotations

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.unified_kv_pool import (
    TpuUnifiedSWAKVPool,
    merge_swa_kv_heads,
    num_swa_groups,
    split_swa_output,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTpuUnifiedSWAKVPoolStorage(unittest.TestCase):
    """Tests for PR 1: shared physical buffers, group page mapping, and head merging."""

    def test_equal_full_and_swa_layers_share_buffers(self):
        pool = TpuUnifiedSWAKVPool(
            size=64,
            size_swa=32,
            page_size=4,
            dtype=torch.bfloat16,
            head_num=4,
            head_dim=128,
            swa_attention_layer_ids=[0, 2],
            full_attention_layer_ids=[1, 3],
            device="cpu",
        )
        self.assertEqual((pool.size, pool.size_swa), (64, 64))
        self.assertEqual((pool.num_swa_groups, pool.num_shared_buffers), (1, 2))
        self.assertFalse(pool.swa_kv_heads_merged)
        self.assertEqual(len(pool.get_state()), 2)
        for full_lid, swa_lid in ((1, 0), (3, 2)):
            self.assertEqual(
                pool.get_fused_kv_buffer(full_lid).data_ptr(),
                pool.get_fused_kv_buffer(swa_lid).data_ptr(),
            )
            self.assertEqual(pool.swa_page_group(swa_lid), 0)
        self.assertEqual(
            pool.get_kv_size_bytes(), pool.full_kv_pool.get_kv_size_bytes()
        )
        self.assertEqual(pool.get_state_buf_infos(), ([], [], []))

    def test_gemma4_ratio_and_head_merging_bf16_and_fp8(self):
        full_ids = [5, 11]
        swa_ids = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                pool = TpuUnifiedSWAKVPool(
                    size=80,
                    size_swa=40,
                    page_size=4,
                    dtype=dtype,
                    head_num=1,
                    head_dim=512,
                    swa_head_num=2,
                    swa_head_dim=256,
                    swa_attention_layer_ids=swa_ids,
                    full_attention_layer_ids=full_ids,
                    device="cpu",
                )
                self.assertTrue(pool.swa_kv_heads_merged)
                self.assertEqual((pool.num_swa_groups, pool.num_shared_buffers), (5, 2))
                self.assertEqual(pool.full_kv_pool.num_blocks, 21)
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

    def test_incompatible_page_shapes_rejected(self):
        with self.assertRaises(ValueError):
            TpuUnifiedSWAKVPool(
                size=32,
                page_size=4,
                dtype=torch.bfloat16,
                head_num=1,
                head_dim=512,
                swa_head_num=3,
                swa_head_dim=128,
                swa_attention_layer_ids=[0, 1],
                full_attention_layer_ids=[2],
                device="cpu",
            )

    def test_merge_swa_kv_heads_and_split_output_exact(self):
        torch.manual_seed(42)
        T, Hq, Hkv, D = 7, 8, 2, 64
        q = torch.randn(T, Hq, D)
        k = torch.randn(T, Hkv, D)
        v = torch.randn(T, Hkv, D)
        scale = D**-0.5

        qg = q.view(T, Hkv, Hq // Hkv, D)
        scores = torch.einsum("thgd,shd->thgs", qg, k) * scale
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        probs = torch.softmax(scores + mask[:, None, None, :], dim=-1)
        out_ref = torch.einsum("thgs,shd->thgd", probs, v).reshape(T, Hq, D)

        q_m, k_m, v_m = merge_swa_kv_heads(q, k, v)
        self.assertEqual(
            (q_m.shape, k_m.shape, v_m.shape), ((T, Hq, 128), (T, 1, 128), (T, 1, 128))
        )
        scores_m = torch.einsum("tqd,skd->tqks", q_m, k_m).squeeze(2) * scale
        probs_m = torch.softmax(scores_m + mask[:, None, :], dim=-1)
        out_m = torch.einsum("tqs,skd->tqd", probs_m, v_m)
        torch.testing.assert_close(
            split_swa_output(out_m, Hkv), out_ref, rtol=1e-5, atol=1e-5
        )

    def test_non_divisible_swa_groups_and_cpu_copy_restore(self):
        self.assertEqual(num_swa_groups(8, 20), 3)
        page_size = 4
        pool = TpuUnifiedSWAKVPool(
            size=48,
            size_swa=48,
            page_size=page_size,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=128,
            swa_attention_layer_ids=list(range(8, 28)),
            full_attention_layer_ids=list(range(8)),
            device="cpu",
        )
        self.assertEqual((pool.num_shared_buffers, pool.num_swa_groups), (8, 3))
        # SWA scattered pages [3, 9, 11] -> [6, 2, 10] across M_swa = 3 groups.
        pool.swa_group_pages[:, 3] = torch.tensor([9, 11], dtype=torch.int32)
        for j in range(20):
            p = 3 if (j // 8) == 0 else int(pool.swa_group_pages[j // 8 - 1, 3])
            pool.get_state()[j % 8][p].fill_(float(10 + j))
        swa_copy = pool.swa_kv_pool.get_cpu_copy(
            torch.arange(3 * page_size, 4 * page_size)
        )
        pool.swa_group_pages[:, 6] = torch.tensor([2, 10], dtype=torch.int32)
        pool.swa_kv_pool.load_cpu_copy(
            swa_copy, torch.arange(6 * page_size, 7 * page_size)
        )
        for j in range(20):
            p = 6 if (j // 8) == 0 else int(pool.swa_group_pages[j // 8 - 1, 6])
            self.assertTrue(torch.all(pool.get_state()[j % 8][p] == float(10 + j)))

        # Full page 2 -> 5 across all 8 shared buffers directly.
        for i in range(8):
            pool.get_state()[i][2].fill_(float(100 + i))
        full_copy = pool.full_kv_pool.get_cpu_copy(
            torch.arange(2 * page_size, 3 * page_size)
        )
        pool.full_kv_pool.load_cpu_copy(
            full_copy, torch.arange(5 * page_size, 6 * page_size)
        )
        for i in range(8):
            self.assertTrue(torch.all(pool.get_state()[i][5] == float(100 + i)))


if __name__ == "__main__":
    unittest.main()



