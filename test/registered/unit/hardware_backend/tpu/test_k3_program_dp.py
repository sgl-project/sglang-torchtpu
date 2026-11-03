"""Attention DP2 on the K3 decode: page regions, block-table ids, and a
token-level golden layout from the DCP32 prefill through the transfer and the
swap to the DCP16 group's reads."""

import itertools
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.k3_programs import sglang_dp as dp

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

RANKS = 32


class TestDpPages(CustomTestCase):
    def test_regions_rows_and_ids(self):
        pages = dp.DpPages(pages=8, num_blocks=20)
        self.assertEqual([pages.group(p) for p in range(1, 9)], [0] * 4 + [1] * 4)
        self.assertEqual(pages.rows(1), (1, 2))
        self.assertEqual(pages.rows(5), (1, 2))  # same rows, the other group's ranks
        self.assertEqual(pages.rows(8), (7, 8))
        self.assertEqual(pages.table_ids(6), (20 + 3, 20 + 4))
        self.assertEqual(list(pages.region(1)), [5, 6, 7, 8])
        with self.assertRaises(ValueError):
            dp.DpPages(pages=7, num_blocks=20)

    def test_table_row_cuts_to_the_touched_blocks(self):
        pages = dp.DpPages(pages=8, num_blocks=20)
        row = pages.table_row([2, 3], seq_len=dp.GROUP_BLOCK_TOKENS + 1, width=6)
        self.assertEqual(row.tolist(), [3, 4, 0, 0, 0, 0])
        row = pages.table_row([6, 7], seq_len=3 * dp.GROUP_BLOCK_TOKENS, width=6)
        self.assertEqual(row.tolist(), [23, 24, 25, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "span DP groups"):
            pages.table_row([2, 6], seq_len=dp.PAGE_TOKENS + 1, width=6)


def _golden(seq_len, group, decode_pages, landing, prefill_rows, num_blocks=24, pages=16):
    """Place a sequence's chunks DCP32 on the prefill, transfer each decode
    rank's rows, swap, and read every chunk back through the DCP16 table."""
    plan = dp.DpPages(pages=pages, num_blocks=num_blocks)
    chunks = -(-seq_len // dp.CHUNK_TOKENS)
    blocks = -(-seq_len // dp.PAGE_TOKENS)
    prefill = np.full((RANKS, num_blocks), -1)
    for k in range(chunks):
        prefill[k % RANKS, prefill_rows[k // RANKS]] = k
    decode = np.full((RANKS, num_blocks), -1)
    # Rows other requests own on the decode: must survive.
    for rank in range(RANKS):
        decode[rank, num_blocks - 1] = 10_000 + rank
    for rank in range(RANKS):
        rows = plan.transfer_rows(rank, decode_pages[:blocks], landing)
        for j, row in enumerate(rows):
            decode[rank, row] = prefill[rank, prefill_rows[j]]
    sources, destinations = plan.swap_rows(decode_pages[:blocks], landing)
    other = 1 - group
    for q in range(dp.GROUP_RANKS):
        for src, dst in zip(sources, destinations):
            decode[group * dp.GROUP_RANKS + q, dst] = decode[other * dp.GROUP_RANKS + q, src]
    width = -(-seq_len // dp.GROUP_BLOCK_TOKENS) + 2
    table = plan.table_row(decode_pages, seq_len, width)
    for k in range(chunks):
        q, b = k % dp.GROUP_RANKS, k // dp.GROUP_RANKS
        ident = int(table[b])
        assert ident // num_blocks == group, (k, ident)
        got = decode[group * dp.GROUP_RANKS + q, ident % num_blocks]
        assert got == k, f"chunk {k}: group rank {q} block {b} row {ident % num_blocks} holds {got}"
    for rank in range(RANKS):
        assert decode[rank, num_blocks - 1] == 10_000 + rank


class TestGoldenLayout(CustomTestCase):
    def test_every_chunk_reads_back_in_either_group(self):
        lengths = (1, 511, 512, 8192, 8193, 16384, 16385, 40000, 65536)
        for group, seq_len in itertools.product((0, 1), lengths):
            with self.subTest(group=group, seq_len=seq_len):
                blocks = -(-seq_len // dp.PAGE_TOKENS)
                plan = dp.DpPages(pages=16, num_blocks=24)
                region, other = list(plan.region(group)), list(plan.region(1 - group))
                decode_pages = region[::-1][: blocks + 1]  # not in order, one spare
                landing = other[1 : 1 + dp.landing_pages_needed(blocks)]
                prefill_rows = [3 + 2 * j for j in range(blocks)]
                _golden(seq_len, group, decode_pages, landing, prefill_rows)

    def test_wrong_landing_group_is_rejected(self):
        plan = dp.DpPages(pages=16, num_blocks=24)
        with self.assertRaisesRegex(ValueError, "landing pages of group 1"):
            plan.transfer_rows(16, [1, 2], [3])


if __name__ == "__main__":
    unittest.main()
