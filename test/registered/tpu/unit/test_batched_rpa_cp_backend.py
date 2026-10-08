# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TPU Pallas Batched RPA CP metadata & LSE merge logic (CPU-runnable)."""

from __future__ import annotations

import unittest

import torch

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit")

try:
    from sglang.test.test_utils import CustomTestCase
except ImportError:
    CustomTestCase = unittest.TestCase

from sglang.srt.hardware_backend.tpu.batched_rpa_cp import (
    build_pcp_batched_metadata,
    ensure_cp_rank_tensor,
    merge_attn_states,
)


class TestBatchedRpaCpBackend(CustomTestCase):
    """CPU unit tests for Batched RPA Context Parallelism components."""

    def test_merge_attn_states(self):
        """Verify that LSE-weighted attention state merging matches mathematical ground truth."""
        batch_size = 2
        num_heads = 4
        head_dim = 128

        # Create synthetic states
        out1 = torch.randn(batch_size, num_heads, head_dim, dtype=torch.float32)
        out2 = torch.randn(batch_size, num_heads, head_dim, dtype=torch.float32)
        lse1 = torch.tensor(
            [[10.0, 12.0, 8.0, 15.0], [5.0, 7.0, 9.0, 11.0]], dtype=torch.float32
        )
        lse2 = torch.tensor(
            [[11.0, 10.0, 9.0, 14.0], [6.0, 6.0, 8.0, 12.0]], dtype=torch.float32
        )

        merged_out, merged_lse = merge_attn_states(out1, lse1, out2, lse2)

        # Reference computation
        expected_lse = torch.log(torch.exp(lse1) + torch.exp(lse2))
        w1 = (torch.exp(lse1) / (torch.exp(lse1) + torch.exp(lse2))).unsqueeze(-1)
        w2 = (torch.exp(lse2) / (torch.exp(lse1) + torch.exp(lse2))).unsqueeze(-1)
        expected_out = out1 * w1 + out2 * w2

        self.assertTrue(torch.allclose(merged_lse, expected_lse, atol=1e-5))
        self.assertTrue(torch.allclose(merged_out, expected_out, atol=1e-5))

    def test_build_pcp_batched_metadata(self):
        """Verify zigzag PCP metadata generation for a 2-rank CP group."""
        cp_size = 2
        page_size = 128
        seq_lens = torch.tensor([512, 1024], dtype=torch.int32)
        prefix_lens = torch.tensor([128, 256], dtype=torch.int32)
        page_indices = torch.arange(16, dtype=torch.int32)
        device = torch.device("cpu")

        # Rank 0
        meta_rank0 = build_pcp_batched_metadata(
            seq_lens=seq_lens,
            prefix_lens=prefix_lens,
            page_indices=page_indices,
            page_size=page_size,
            cp_size=cp_size,
            cp_rank=0,
            device=device,
        )

        # Rank 0 owns chunk 0 and chunk 2P-1 = 3
        self.assertEqual(meta_rank0.num_reqs, 2)
        # 2 chunks per request -> 4 entries + initial 0 = 5 entries in pcp_cu
        self.assertEqual(meta_rank0.pcp_cu_q_lens.shape[0], 5)
        # Tail chunks should have update_kv_cache = True, head chunks = False
        expected_write = torch.tensor([False, True, False, True], dtype=torch.bool)
        self.assertTrue(torch.equal(meta_rank0.update_kv_cache, expected_write))

        # Check striped page indices: strided pages owned by rank 0
        expected_striped = torch.tensor(
            [0, 2, 4, 6, 0, 0, 0, 0, 8, 10, 12, 14, 0, 0, 0, 0], dtype=torch.int32
        )
        self.assertTrue(torch.equal(meta_rank0.striped_page_indices, expected_striped))

    def test_ensure_cp_rank_tensor(self):
        """Verify persistent rank tensor creation."""
        device = torch.device("cpu")
        t1 = ensure_cp_rank_tensor(device, cp_rank=1)
        t2 = ensure_cp_rank_tensor(device, cp_rank=1)
        self.assertIs(t1, t2)
        self.assertEqual(t1.item(), 1)

    def test_dcp_decode_state_merge(self):
        """Verify mathematical equivalence of DCP decode multi-rank state merge."""
        cp_size = 2
        num_reqs = 3
        num_heads = 4
        head_dim = 16

        curr_lse = torch.randn(num_reqs, num_heads)
        curr_out = torch.randn(num_reqs, num_heads, head_dim)

        context_lse_r0 = torch.randn(num_reqs, num_heads)
        context_out_r0 = torch.randn(num_reqs, num_heads, head_dim)
        context_lse_r1 = torch.randn(num_reqs, num_heads)
        context_out_r1 = torch.randn(num_reqs, num_heads, head_dim)

        # Simulate all_gather along dim 0
        all_context_lse = torch.cat([context_lse_r0, context_lse_r1], dim=0)
        all_context_lse_3d = all_context_lse.view(cp_size, -1, *curr_lse.shape[1:])

        stacked_lses = torch.cat([curr_lse.unsqueeze(0), all_context_lse_3d], dim=0)
        self.assertEqual(stacked_lses.shape, (1 + cp_size, num_reqs, num_heads))

        global_max_lse = torch.amax(stacked_lses, dim=0)
        exp_all = torch.exp(stacked_lses - global_max_lse.unsqueeze(0))
        sum_exp = torch.sum(exp_all, dim=0)

        # Reference manual merge across all 3 components (curr, r0, r1)
        w_curr = torch.exp(curr_lse - global_max_lse).unsqueeze(-1)
        w_r0 = torch.exp(context_lse_r0 - global_max_lse).unsqueeze(-1)
        w_r1 = torch.exp(context_lse_r1 - global_max_lse).unsqueeze(-1)
        expected_out = (
            curr_out * w_curr + context_out_r0 * w_r0 + context_out_r1 * w_r1
        ) / sum_exp.unsqueeze(-1)

        # Simulated DCP pipeline:
        scale_r0 = torch.exp(context_lse_r0 - global_max_lse).unsqueeze(-1)
        scale_r1 = torch.exp(context_lse_r1 - global_max_lse).unsqueeze(-1)
        sum_scaled_context = context_out_r0 * scale_r0 + context_out_r1 * scale_r1
        scale_curr = torch.exp(curr_lse - global_max_lse).unsqueeze(-1)
        total_out = sum_scaled_context + curr_out * scale_curr
        merged_out = total_out / sum_exp.unsqueeze(-1)

        self.assertTrue(torch.allclose(merged_out, expected_out, atol=1e-5))

        # Test robustness against -inf LSE and uninitialized NaN output buffers
        context_lse_r1_inf = context_lse_r1.clone()
        context_out_r1_nan = context_out_r1.clone()
        context_lse_r1_inf[0, 0] = float("-inf")
        context_out_r1_nan[0, 0] = float("nan")

        all_lse_inf = torch.cat([context_lse_r0, context_lse_r1_inf], dim=0).view(
            cp_size, -1, *curr_lse.shape[1:]
        )
        stacked_inf = torch.cat([curr_lse.unsqueeze(0), all_lse_inf], dim=0)
        max_inf = torch.amax(stacked_inf, dim=0)
        max_inf_safe = torch.where(
            torch.isinf(max_inf), torch.zeros_like(max_inf), max_inf
        )

        scale_r0_safe = torch.where(
            torch.isinf(context_lse_r0),
            torch.zeros_like(context_lse_r0),
            torch.exp(context_lse_r0 - max_inf_safe),
        ).unsqueeze(-1)
        scale_r1_safe = torch.where(
            torch.isinf(context_lse_r1_inf),
            torch.zeros_like(context_lse_r1_inf),
            torch.exp(context_lse_r1_inf - max_inf_safe),
        ).unsqueeze(-1)
        scale_curr_safe = torch.where(
            torch.isinf(curr_lse),
            torch.zeros_like(curr_lse),
            torch.exp(curr_lse - max_inf_safe),
        ).unsqueeze(-1)

        sum_scaled = (
            torch.nan_to_num(context_out_r0, nan=0.0) * scale_r0_safe
            + torch.nan_to_num(context_out_r1_nan, nan=0.0) * scale_r1_safe
        )
        total = sum_scaled + torch.nan_to_num(curr_out, nan=0.0) * scale_curr_safe
        exp_sum = torch.sum(torch.exp(stacked_inf - max_inf_safe.unsqueeze(0)), dim=0)
        merged_safe = total / exp_sum.unsqueeze(-1)

        self.assertFalse(torch.isnan(merged_safe).any())
        self.assertFalse(torch.isinf(merged_safe).any())


if __name__ == "__main__":
    unittest.main()
