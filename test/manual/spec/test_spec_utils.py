import unittest

import numpy as np
import torch

from sglang.srt.mem_cache.memory_pool import copy_all_layer_kv_cache_tiled
from sglang.srt.speculative.spec_utils import (
    _assign_draft_cache_locs_torch,
    _assign_req_to_token_pool_torch,
    _create_extend_after_decode_spec_info_torch,
    assign_draft_cache_locs,
)
from sglang.srt.utils import next_power_of_2

BYTES_PER_TILE = 128


class TestSpecUtils(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_ptrs = torch.zeros(2, 1, dtype=torch.uint64, device=self.device)
        self.k_cache = [
            torch.zeros((100, 1, 1), dtype=torch.float32, device=self.device)
        ]
        self.v_cache = [
            torch.zeros((100, 1, 1), dtype=torch.float32, device=self.device)
        ]
        self.k_cache[0][:11, 0, 0] = torch.tensor(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            dtype=torch.float32,
            device=self.device,
        )
        self.v_cache[0][:11, 0, 0] = torch.tensor(
            [-0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0],
            dtype=torch.float32,
            device=self.device,
        )
        self.data_ptrs[0, 0] = self.k_cache[0].data_ptr()
        self.data_ptrs[1, 0] = self.v_cache[0].data_ptr()

        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_cache + self.v_cache
            ],
            device=self.device,
            dtype=torch.int64,
        )

    def test_assign_draft_cache_locs_single_seq(self):
        # Testing Setup: req_to_token starting from 4
        # 4,5,6,7,{8,9,10}, 8,9,10 is the last partial page, 3 tokens < page_size=4
        # next kv cache will be stored starting 11,12,13...
        device = self.device
        num_seqs = 1
        page_size = 4
        speculative_num_steps = 5
        topk = 8
        seq_lens_num = 7
        extend_lens_num = 61  # includes the duplicated last page
        req_pool_indices = torch.arange(num_seqs, dtype=torch.int32, device=device)
        req_to_token = torch.zeros((num_seqs, 100), dtype=torch.int32, device=device)
        req_to_token[0, :seq_lens_num] = torch.tensor(
            [4, 5, 6, 7, 8, 9, 10], device=device
        )
        seq_lens = torch.tensor([seq_lens_num], dtype=torch.int32, device=device)
        extend_lens = torch.tensor([extend_lens_num], dtype=torch.int32, device=device)
        num_new_pages_per_topk = torch.tensor([2], dtype=torch.int32, device=device)
        out_cache_loc = torch.arange(11, 11 + extend_lens_num, device=device)
        last_page_lens = torch.tensor([3], dtype=torch.int32, device=device)
        last_page_lens_cumsum = torch.cumsum(last_page_lens, dim=0)
        duplicate_cache_len = last_page_lens.sum().item() * (topk - 1)
        target_cache_loc = torch.zeros(
            duplicate_cache_len, dtype=torch.int32, device=device
        )
        source_cache_loc = torch.zeros(
            duplicate_cache_len, dtype=torch.int32, device=device
        )
        assign_draft_cache_locs[(num_seqs,)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            extend_lens,
            num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            req_to_token.shape[1],
            topk,
            speculative_num_steps,
            page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(speculative_num_steps + page_size),
        )

        out_cache_loc = out_cache_loc[: num_seqs * topk * speculative_num_steps]
        expected_source_cache_loc = torch.tensor(
            [8, 9, 10] * (topk - 1), device=device, dtype=torch.int32
        )
        assert torch.allclose(source_cache_loc, expected_source_cache_loc)

        copy_all_layer_kv_cache_tiled[(len(self.data_ptrs),)](
            self.data_ptrs,
            self.data_strides,
            target_cache_loc,
            source_cache_loc,
            len(target_cache_loc),
            next_power_of_2(len(target_cache_loc)),
            BYTES_PER_TILE,
        )
        assert torch.allclose(
            self.k_cache[0][16:19, 0, 0],
            torch.tensor(
                [0.8, 0.9, 1.0],
                dtype=torch.float32,
                device=device,
            ),
        )
        assert torch.allclose(
            self.v_cache[0][16:19, 0, 0],
            torch.tensor(
                [-0.8, -0.9, -1.0],
                dtype=torch.float32,
                device=device,
            ),
        )

    def test_assign_draft_cache_locs_multi_seq(self):
        device = self.device
        num_seqs = 3
        page_size = 4
        speculative_num_steps = 5
        topk = 8
        req_pool_indices = torch.arange(num_seqs, dtype=torch.int32, device=device)
        req_to_token = torch.zeros((num_seqs, 100), dtype=torch.int32, device=device)
        seq_lens = torch.tensor([8, 7, 5], dtype=torch.int32, device=device)
        extend_lens = torch.tensor([64, 64, 64], dtype=torch.int32, device=device)
        num_new_pages_per_topk = torch.tensor(
            [2, 2, 2], dtype=torch.int32, device=device
        )
        req_to_token = torch.zeros((num_seqs, 100), dtype=torch.int32, device=device)
        req_to_token[0, :8] = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11], device=device)
        req_to_token[1, :7] = torch.tensor([4, 5, 6, 7, 8, 9, 10], device=device)
        req_to_token[2, :5] = torch.tensor([4, 5, 6, 7, 8], device=device)
        last_page_lens = torch.tensor([0, 3, 1], dtype=torch.int32, device=device)
        last_page_lens_cumsum = torch.cumsum(last_page_lens, dim=0)
        duplicate_cache_len = last_page_lens.sum().item() * (topk - 1)
        out_cache_loc = torch.arange(
            12, 12 + torch.sum(extend_lens), dtype=torch.int32, device=device
        )
        target_cache_loc = torch.zeros(
            duplicate_cache_len, dtype=torch.int32, device=device
        )
        source_cache_loc = torch.zeros(
            duplicate_cache_len, dtype=torch.int32, device=device
        )
        assign_draft_cache_locs[(num_seqs,)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            extend_lens,
            num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            req_to_token.shape[1],
            topk,
            speculative_num_steps,
            page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(speculative_num_steps + page_size),
        )
        out_cache_loc = out_cache_loc[: num_seqs * topk * speculative_num_steps]
        # fmt: off
        expected_out_cache_loc = torch.tensor([
            12,  13,  14,  15,  16,
            20,  21,  22,  23,  24,
            28,  29,  30,  31,  32,
            36,  37,  38,  39,  40,
            44,  45,  46,  47,  48,
            52,  53,  54,  55,  56,
            60,  61,  62,  63,  64,
            68,  69,  70,  71,  72,
            76,  77,  78,  79,  80,
            84,  85,  86,  87,  88,
            92,  93,  94,  95,  96,
            100, 101, 102, 103, 104,
            108, 109, 110, 111, 112,
            116, 117, 118, 119, 120,
            124, 125, 126, 127, 128,
            132, 133, 134, 135, 136,
            140, 141, 142, 143, 144,
            148, 149, 150, 151, 152,
            156, 157, 158, 159, 160,
            164, 165, 166, 167, 168,
            172, 173, 174, 175, 176,
            180, 181, 182, 183, 184,
            188, 189, 190, 191, 192,
            196, 197, 198, 199, 200
        ], device=device, dtype=torch.int32)
        expected_source_cache_loc = torch.tensor([8, 9, 10] * 7 + [8] * 7, device=device, dtype=torch.int32)
        expected_target_cache_loc = torch.tensor([
            81,  82,  83,  89,  90,  91,  97,  98,  99, 105, 106, 107, 113, 114,
           115, 121, 122, 123, 129, 130, 131, 147, 155, 163, 171, 179, 187, 195
        ], device=device, dtype=torch.int32)
        # fmt: on
        assert torch.allclose(out_cache_loc, expected_out_cache_loc)
        assert torch.allclose(source_cache_loc, expected_source_cache_loc)
        assert torch.allclose(target_cache_loc, expected_target_cache_loc)
        copy_all_layer_kv_cache_tiled[(len(self.data_ptrs),)](
            self.data_ptrs,
            self.data_strides,
            target_cache_loc,
            source_cache_loc,
            len(target_cache_loc),
            next_power_of_2(len(target_cache_loc)),
            BYTES_PER_TILE,
        )
        assert torch.allclose(
            self.k_cache[0][81:84, 0, 0],
            torch.tensor(
                [0.8, 0.9, 1.0],
                dtype=torch.float32,
                device=device,
            ),
        )
        assert torch.allclose(
            self.v_cache[0][81:84, 0, 0],
            torch.tensor(
                [-0.8, -0.9, -1.0],
                dtype=torch.float32,
                device=device,
            ),
        )

    def test_assign_draft_cache_locs_page_size_1(self):
        # Test to make sure page_size=1 not affected
        device = self.device
        num_seqs = 1
        page_size = 1
        speculative_num_steps = 5
        topk = 8
        seq_lens_num = 7
        extend_lens_num = topk * speculative_num_steps
        req_pool_indices = torch.arange(num_seqs, dtype=torch.int32, device=device)
        req_to_token = torch.zeros((num_seqs, 100), dtype=torch.int32, device=device)
        req_to_token[0, :seq_lens_num] = torch.tensor(
            [4, 5, 6, 7, 8, 9, 10], device=device
        )
        seq_lens = torch.tensor([seq_lens_num], dtype=torch.int32, device=device)
        extend_lens = torch.tensor([extend_lens_num], dtype=torch.int32, device=device)
        num_new_pages_per_topk = torch.tensor([2], dtype=torch.int32, device=device)
        out_cache_loc = torch.arange(11, 11 + extend_lens_num, device=device)
        last_page_lens = torch.tensor([3], dtype=torch.int32, device=device)
        duplicate_cache_len = 0
        target_cache_loc = None
        source_cache_loc = None
        last_page_lens_cumsum = None
        assign_draft_cache_locs[(num_seqs,)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            extend_lens,
            num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            req_to_token.shape[1],
            topk,
            speculative_num_steps,
            page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(speculative_num_steps + page_size),
        )
        out_cache_loc = out_cache_loc[: num_seqs * topk * speculative_num_steps]
        expected_out_cache_loc = torch.arange(11, 11 + extend_lens_num, device=device)
        assert torch.allclose(out_cache_loc, expected_out_cache_loc)

    def test_assign_draft_cache_locs_page_size_gt_spec_steps(self):
        device = self.device
        num_seqs = 1
        page_size = 16
        speculative_num_steps = 4
        topk = 3
        seq_lens_num = 12
        pool_len = 256
        req_pool_indices = torch.arange(num_seqs, dtype=torch.int32, device=device)
        req_to_token = torch.zeros(
            (num_seqs, pool_len), dtype=torch.int32, device=device
        )
        req_to_token[0, :seq_lens_num] = torch.arange(
            seq_lens_num, dtype=torch.int32, device=device
        )
        seq_lens = torch.tensor([seq_lens_num], dtype=torch.int32, device=device)
        last_page_len = seq_lens_num % page_size
        last_page_lens = torch.tensor([last_page_len], dtype=torch.int32, device=device)
        last_page_lens_cumsum = torch.cumsum(last_page_lens, dim=0)
        num_new_pages_per_topk_val = (
            last_page_len + speculative_num_steps + page_size - 1
        ) // page_size
        num_new_pages_per_topk = torch.tensor(
            [num_new_pages_per_topk_val], dtype=torch.int32, device=device
        )
        extend_lens_num = num_new_pages_per_topk_val * page_size * topk
        extend_lens = torch.tensor([extend_lens_num], dtype=torch.int32, device=device)
        out_cache_loc = torch.arange(
            2000, 2000 + extend_lens_num, dtype=torch.int32, device=device
        )
        duplicate_cache_len = last_page_lens.sum().item() * (topk - 1)
        target_cache_loc = torch.zeros(
            duplicate_cache_len, dtype=torch.int32, device=device
        )
        source_cache_loc = torch.zeros(
            duplicate_cache_len, dtype=torch.int32, device=device
        )
        assign_draft_cache_locs[(num_seqs,)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            extend_lens,
            num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            req_to_token.shape[1],
            topk,
            speculative_num_steps,
            page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(speculative_num_steps + page_size),
        )
        trimmed = out_cache_loc[: num_seqs * topk * speculative_num_steps]
        expected = []
        for topk_id in range(topk):
            start = seq_lens_num + topk_id * num_new_pages_per_topk_val * page_size
            expected.append(
                req_to_token[0, start : start + speculative_num_steps].clone()
            )
        expected_out_cache_loc = torch.cat(expected)
        assert torch.allclose(trimmed, expected_out_cache_loc)


class TestAssignDraftCacheLocs(unittest.TestCase):
    """Verify _assign_draft_cache_locs_torch against a naive reference."""

    @staticmethod
    def _ref(
        req_pool_indices,
        req_to_token,
        seq_lens,
        out_cache_loc,
        topk,
        speculative_num_steps,
        batch_size,
    ):
        """Naive per-request loop for page_size=1 or topk=1."""
        copy_len = topk * speculative_num_steps
        for pid in range(batch_size):
            out_start = pid * copy_len
            kv_start = int(seq_lens[pid].item())
            row = int(req_pool_indices[pid].item())
            for i in range(copy_len):
                req_to_token[row, kv_start + i] = out_cache_loc[out_start + i]

    def test_page_size_1_single(self):
        """page_size=1, topk=1: simplest path."""
        device = "cpu"
        bs, pool_len, topk, steps = 1, 100, 1, 5
        req_pool_indices = torch.tensor([0], dtype=torch.int32, device=device)
        req_to_token = torch.zeros((bs, pool_len), dtype=torch.int32, device=device)
        req_to_token[0, :7] = torch.tensor([4, 5, 6, 7, 8, 9, 10], device=device)
        seq_lens = torch.tensor([7], dtype=torch.int32, device=device)
        out_cache_loc = torch.arange(11, 11 + steps, device=device)

        ref_table = req_to_token.clone()
        ref_out = out_cache_loc.clone()
        self._ref(req_pool_indices, ref_table, seq_lens, ref_out, topk, steps, bs)

        got_table = req_to_token.clone()
        got_out = out_cache_loc.clone()
        _assign_draft_cache_locs_torch(
            req_pool_indices,
            got_table,
            seq_lens,
            None,
            None,
            got_out,
            None,
            None,
            None,
            0,
            pool_len,
            topk,
            steps,
            1,
            bs,
        )

        torch.testing.assert_close(got_table, ref_table)
        torch.testing.assert_close(got_out, ref_out)

    def test_page_size_1_batch(self):
        """page_size=1, topk=8, batch of 3."""
        device = "cpu"
        bs, pool_len, topk, steps = 3, 100, 8, 5
        req_pool_indices = torch.arange(bs, dtype=torch.int32, device=device)
        req_to_token = torch.zeros((bs, pool_len), dtype=torch.int32, device=device)
        seq_lens = torch.tensor([8, 7, 5], dtype=torch.int32, device=device)
        total = bs * topk * steps
        out_cache_loc = torch.arange(100, 100 + total, device=device)

        ref_table = req_to_token.clone()
        ref_out = out_cache_loc.clone()
        self._ref(req_pool_indices, ref_table, seq_lens, ref_out, topk, steps, bs)

        got_table = req_to_token.clone()
        got_out = out_cache_loc.clone()
        _assign_draft_cache_locs_torch(
            req_pool_indices,
            got_table,
            seq_lens,
            None,
            None,
            got_out,
            None,
            None,
            None,
            0,
            pool_len,
            topk,
            steps,
            1,
            bs,
        )

        torch.testing.assert_close(got_table, ref_table)
        torch.testing.assert_close(got_out, ref_out)

    def test_empty_batch(self):
        device = "cpu"
        table = torch.full((4, 32), -1, dtype=torch.int32, device=device)
        expected = table.clone()
        out = torch.zeros(0, dtype=torch.int32, device=device)
        _assign_draft_cache_locs_torch(
            torch.zeros(0, dtype=torch.int32, device=device),
            table,
            torch.zeros(0, dtype=torch.int32, device=device),
            None,
            None,
            out,
            None,
            None,
            None,
            0,
            32,
            1,
            5,
            1,
            0,
        )
        torch.testing.assert_close(table, expected)

    def test_oversized_inputs(self):
        """Input tensors longer than batch_size; trailing entries must be ignored."""
        device = "cpu"
        bs, pool_len, topk, steps = 2, 100, 1, 5
        copy_len = topk * steps
        # 2 active requests + 1 trailing garbage entry.
        req_pool_indices = torch.tensor([0, 1, 99], dtype=torch.int32, device=device)
        req_to_token = torch.zeros((3, pool_len), dtype=torch.int32, device=device)
        seq_lens = torch.tensor([7, 3, 999], dtype=torch.int32, device=device)
        # out_cache_loc has room for 3 requests but only 2 should be used.
        out_cache_loc = torch.arange(
            100, 100 + 3 * copy_len, device=device, dtype=torch.int32
        )

        ref_table = req_to_token.clone()
        self._ref(
            req_pool_indices[:bs],
            ref_table,
            seq_lens[:bs],
            out_cache_loc[: bs * copy_len],
            topk,
            steps,
            bs,
        )

        got_table = req_to_token.clone()
        _assign_draft_cache_locs_torch(
            req_pool_indices,
            got_table,
            seq_lens,
            None,
            None,
            out_cache_loc,
            None,
            None,
            None,
            0,
            pool_len,
            topk,
            steps,
            1,
            bs,
        )

        torch.testing.assert_close(got_table, ref_table)

    def test_minimal_copy_len(self):
        """topk=1, steps=1: single slot written per request."""
        device = "cpu"
        bs, pool_len, topk, steps = 2, 64, 1, 1
        copy_len = topk * steps  # 1
        req_pool_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
        req_to_token = torch.zeros((bs, pool_len), dtype=torch.int32, device=device)
        seq_lens = torch.tensor([10, 20], dtype=torch.int32, device=device)
        out_cache_loc = torch.tensor([77, 88], dtype=torch.int32, device=device)

        ref_table = req_to_token.clone()
        self._ref(req_pool_indices, ref_table, seq_lens, out_cache_loc, topk, steps, bs)

        got_table = req_to_token.clone()
        _assign_draft_cache_locs_torch(
            req_pool_indices,
            got_table,
            seq_lens,
            None,
            None,
            out_cache_loc,
            None,
            None,
            None,
            0,
            pool_len,
            topk,
            steps,
            1,
            bs,
        )

        torch.testing.assert_close(got_table, ref_table)

    def test_unsupported_page_size_topk(self):
        """page_size>1 and topk>1 must raise NotImplementedError."""
        device = "cpu"
        bs, pool_len, topk, steps, page_size = 1, 64, 2, 3, 4
        req_pool_indices = torch.tensor([0], dtype=torch.int32, device=device)
        req_to_token = torch.zeros((1, pool_len), dtype=torch.int32, device=device)
        seq_lens = torch.tensor([5], dtype=torch.int32, device=device)
        out_cache_loc = torch.arange(100, 100 + topk * steps, dtype=torch.int32)

        with self.assertRaises(NotImplementedError):
            _assign_draft_cache_locs_torch(
                req_pool_indices,
                req_to_token,
                seq_lens,
                None,
                None,
                out_cache_loc,
                None,
                None,
                None,
                0,
                pool_len,
                topk,
                steps,
                page_size,
                bs,
            )


class TestAssignReqToTokenPool(unittest.TestCase):
    """Verify _assign_req_to_token_pool_torch against a naive reference loop."""

    @staticmethod
    def _ref(req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc):
        pos = 0
        for i in range(len(req_pool_indices)):
            row = int(req_pool_indices[i])
            for j in range(int(start_offset[i]), int(end_offset[i])):
                req_to_token[row, j] = out_cache_loc[pos]
                pos += 1

    def _run_case(self, batch_size, pool_len, max_tokens, seed=0):
        torch.manual_seed(seed)
        num_rows = batch_size + 10
        req_pool_indices = torch.randperm(num_rows)[:batch_size]
        start_offset = torch.randint(0, pool_len - max_tokens, (batch_size,))
        lengths = torch.randint(1, max_tokens + 1, (batch_size,))
        end_offset = start_offset + lengths
        out_cache_loc = torch.randint(
            0, 10000, (int(lengths.sum()),), dtype=torch.int64
        )

        ref = torch.full((num_rows, pool_len), -1, dtype=torch.int64)
        self._ref(req_pool_indices, ref, start_offset, end_offset, out_cache_loc)

        got = torch.full((num_rows, pool_len), -1, dtype=torch.int64)
        _assign_req_to_token_pool_torch(
            req_pool_indices, got, start_offset, end_offset, out_cache_loc, batch_size
        )
        torch.testing.assert_close(got, ref)

    def test_single_request(self):
        self._run_case(batch_size=1, pool_len=128, max_tokens=10)

    def test_batch(self):
        self._run_case(batch_size=16, pool_len=256, max_tokens=31)

    def test_large_start_offset(self):
        """Regression: dst_index must be clamped to avoid OOB."""
        self._run_case(batch_size=8, pool_len=64, max_tokens=15, seed=42)

    def test_empty_batch(self):
        table = torch.full((4, 32), -1, dtype=torch.int64)
        expected = table.clone()
        _assign_req_to_token_pool_torch(
            torch.zeros(0, dtype=torch.long),
            table,
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.int64),
            0,
        )
        torch.testing.assert_close(table, expected)


class TestCreateExtendAfterDecodeSpecInfo(unittest.TestCase):
    """Verify _create_extend_after_decode_spec_info_torch against a naive reference."""

    @staticmethod
    def _ref(accept_tokens, seq_lens, accept_lens, positions, bonus_tokens):
        """Naive per-request loop matching the Triton kernel."""
        offset = 0
        for pid in range(len(seq_lens)):
            seq_length = int(seq_lens[pid].item())
            accept_len = int(accept_lens[pid].item())
            for j in range(accept_len):
                positions[offset + j] = seq_length - accept_len + j
            bonus_tokens[pid] = accept_tokens[offset + accept_len - 1]
            offset += accept_len

    def _run_case(self, seq_vals, accept_vals):
        device = "cpu"
        bs = len(seq_vals)
        seq_lens = torch.tensor(seq_vals, dtype=torch.int32, device=device)
        accept_lens = torch.tensor(accept_vals, dtype=torch.int32, device=device)
        total_tokens = sum(accept_vals)
        accept_tokens = torch.arange(
            100, 100 + total_tokens, dtype=torch.int32, device=device
        )

        ref_pos = torch.zeros(total_tokens, dtype=torch.long, device=device)
        ref_bonus = torch.zeros(bs, dtype=torch.int32, device=device)
        self._ref(accept_tokens, seq_lens, accept_lens, ref_pos, ref_bonus)

        got_pos = torch.zeros(total_tokens, dtype=torch.long, device=device)
        got_bonus = torch.zeros(bs, dtype=torch.int32, device=device)
        _create_extend_after_decode_spec_info_torch(
            accept_tokens,
            seq_lens,
            accept_lens,
            got_pos,
            got_bonus,
        )

        torch.testing.assert_close(got_pos, ref_pos)
        torch.testing.assert_close(got_bonus, ref_bonus)

    def test_single_request_all_accepted(self):
        """1 request, accepted 6 tokens (5 drafts + 1 bonus)."""
        self._run_case(seq_vals=[106], accept_vals=[6])

    def test_single_request_one_accepted(self):
        """1 request, only bonus token accepted."""
        self._run_case(seq_vals=[50], accept_vals=[1])

    def test_batch_uniform(self):
        """3 requests, all accepted same number."""
        self._run_case(seq_vals=[20, 30, 40], accept_vals=[3, 3, 3])

    def test_batch_varying(self):
        """3 requests with different accept counts."""
        self._run_case(seq_vals=[10, 20, 30], accept_vals=[1, 4, 2])

    def test_empty_batch(self):
        device = "cpu"
        positions = torch.zeros(0, dtype=torch.long, device=device)
        bonus = torch.zeros(0, dtype=torch.int32, device=device)
        _create_extend_after_decode_spec_info_torch(
            torch.zeros(0, dtype=torch.int32, device=device),
            torch.zeros(0, dtype=torch.int32, device=device),
            torch.zeros(0, dtype=torch.int32, device=device),
            positions,
            bonus,
        )
        assert positions.numel() == 0
        assert bonus.numel() == 0

    def test_large_batch(self):
        """32 requests with random accept counts (stress test for vectorized path)."""
        torch.manual_seed(7)
        bs = 32
        seq_vals = torch.randint(10, 200, (bs,)).tolist()
        accept_vals = torch.randint(1, 7, (bs,)).tolist()
        self._run_case(seq_vals, accept_vals)

    def test_all_accept_one(self):
        """All requests accept exactly 1 token (bonus-only, no position range)."""
        self._run_case(seq_vals=[5, 10, 15, 20], accept_vals=[1, 1, 1, 1])

    def test_all_accept_max(self):
        """All requests accept the maximum number of tokens."""
        self._run_case(seq_vals=[50, 60, 70], accept_vals=[6, 6, 6])


if __name__ == "__main__":
    unittest.main()
