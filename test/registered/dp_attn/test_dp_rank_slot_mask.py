# SPDX-License-Identifier: Apache-2.0
"""Index-free DP slot placement must match the index form exactly.

The TPU attention-TP > 1 path builds a rank's slot in the MAX_LEN global DP
buffer with a broadcast 0/1 mask instead of a runtime-index scatter /
index_select, because the index form core-halts tpu7x (e0200
RuntimeUnexpectedCoreHalt) once the buffer reaches [dp_size * 512, hidden].
The substitution is only safe if it is bit-identical, which is what this
checks. Pure tensor arithmetic, so it runs on CPU without a TPU.
"""

from __future__ import annotations

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

# Pure tensor arithmetic: it exercises a TPU-only code path but needs no
# accelerator, so it runs on the cheap CPU runners.
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.layers import dp_attention


class TestDpRankSlotMask(unittest.TestCase):
    def setUp(self):
        self._saved = dict(dp_attention._RANK_TENSORS)

    def tearDown(self):
        dp_attention._RANK_TENSORS.clear()
        dp_attention._RANK_TENSORS.update(self._saved)

    @staticmethod
    def _set_attn_dp_rank(rank: int, device: torch.device) -> None:
        # Mirror _init_rank_tensors: a pre-existing 0-D tensor, so the rank
        # stays a runtime operand rather than a graph constant.
        dp_attention._RANK_TENSORS[("attn_dp", device)] = torch.tensor(
            rank, device=device, dtype=torch.int64
        )

    def _check(self, dp_size: int, rank: int, rows: int, hidden: int, dtype):
        device = torch.device("cpu")
        self._set_attn_dp_rank(rank, device)
        local = torch.randn(rows, hidden).to(dtype)
        glob = torch.randn(dp_size * rows, hidden).to(dtype)

        idx = dp_attention._dp_rank_slot_rows(num_rows=rows, device=device)
        sel = dp_attention._dp_rank_slot_mask(dp_size, device)

        self.assertEqual(sel.shape, (dp_size,))
        self.assertEqual(sel.dtype, torch.bool)
        self.assertEqual(sel.sum().item(), 1)
        self.assertEqual(sel.to(torch.int64).argmax().item(), rank)

        zero = torch.zeros((), dtype=dtype)

        # Gather side: zeros_like + scatter  ==  where(mask, tiled, 0).
        want = torch.zeros_like(glob)
        want[idx] = local
        got = torch.where(sel.view(dp_size, 1, 1), local.unsqueeze(0), zero).reshape(
            glob.shape
        )
        self.assertTrue(torch.equal(want, got))
        self.assertEqual(got.dtype, want.dtype)

        # Scatter side: index_select  ==  masked sum over the slot axis.
        want = glob.index_select(0, idx)
        got = torch.where(
            sel.view(dp_size, 1, 1), glob.view(dp_size, rows, hidden), zero
        ).sum(0)
        self.assertTrue(torch.equal(want, got))
        self.assertEqual(got.dtype, want.dtype)

    def test_every_rank_of_several_dp_sizes(self):
        for dp_size in (1, 2, 4, 8):
            for rank in range(dp_size):
                with self.subTest(dp_size=dp_size, rank=rank):
                    self._check(dp_size, rank, rows=16, hidden=8, dtype=torch.float32)

    def test_bfloat16_and_the_shape_that_halts(self):
        # 512 rows per slot is where the scatter form starts core-halting.
        self._check(4, 2, rows=512, hidden=64, dtype=torch.bfloat16)

    def test_single_row_slot(self):
        # Decode batches shrink to one row per rank at the tail of a run.
        self._check(4, 3, rows=1, hidden=8, dtype=torch.float32)

    def test_inf_in_one_slot_does_not_poison_the_others(self):
        # A 0/1 *scale* would make 0 * inf = NaN in every other rank's slot,
        # and the all-reduce would then spread it to all ranks. Selecting with
        # torch.where writes exact zeros, matching the index form it replaces.
        device = torch.device("cpu")
        dp_size, rows, hidden, rank = 4, 8, 4, 1
        self._set_attn_dp_rank(rank, device)
        local = torch.full((rows, hidden), float("inf"))
        sel = dp_attention._dp_rank_slot_mask(dp_size, device)

        got = torch.where(
            sel.view(dp_size, 1, 1), local.unsqueeze(0), torch.zeros(())
        ).reshape(dp_size * rows, hidden)

        idx = dp_attention._dp_rank_slot_rows(num_rows=rows, device=device)
        want = torch.zeros(dp_size * rows, hidden)
        want[idx] = local
        self.assertFalse(torch.isnan(got).any())
        self.assertTrue(torch.equal(want, got))


if __name__ == "__main__":
    unittest.main()
