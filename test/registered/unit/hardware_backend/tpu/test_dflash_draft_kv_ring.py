"""DraftKvRing slot math: the ring the DFlash draft reads must hold what it wrote."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.experimental.speculative.dflash.draft_kv_ring import DraftKvRing

_WINDOW, _WIDTH, _PAGE = 1024, 8, 128


class TestDraftKvRing(CustomTestCase):
    def setUp(self):
        self.ring = DraftKvRing.create(
            window=_WINDOW, verify_width=_WIDTH, page_size=_PAGE
        )
        self.table = self.ring.req_to_token(num_req_rows=6, width=20_000)

    def test_reader_table_matches_writer_slots(self):
        """The draft attention pages through req_to_token while verify writes
        at verify_candidates: both must name the same slot for a position,
        and each logical page must be one contiguous ring page."""
        committed = torch.tensor([0, 1275, 5000, 19_000 - 2 * _WIDTH])
        rows = torch.tensor([1, 2, 3, 5])
        cand = self.ring.verify_candidates(
            req_rows=rows, committed_lens=committed, num_candidates=2 * _WIDTH
        )
        for i, (row, base) in enumerate(zip(rows.tolist(), committed.tolist())):
            want = self.table[row, base : base + 2 * _WIDTH].to(torch.int64)
            self.assertEqual(cand[i].tolist(), want.tolist())
        for page in (0, 7, 9, 10, 11, 150):
            slots = self.table[3, page * _PAGE : (page + 1) * _PAGE].to(torch.int64)
            self.assertEqual(int(slots[0]) % _PAGE, 0, page)
            self.assertTrue(bool((slots.diff() == 1).all()), page)

    def test_window_never_aliases(self):
        """A verify reads up to window + width positions back while writing
        width new ones: no two of them may share a slot, and request rows and
        the trash never overlap."""
        ring = self.ring.ring_tokens
        self.assertGreaterEqual(ring, _WINDOW + 2 * _WIDTH)
        for end in (ring, 5000, 19_999):
            span = self.table[2, end - ring : end]
            self.assertEqual(span.unique().numel(), ring, end)
        lows = self.table.min(dim=1).values.tolist()
        highs = self.table.max(dim=1).values.tolist()
        self.assertGreaterEqual(lows[0], self.ring.trash_tokens)
        for row in range(1, 6):
            self.assertGreater(lows[row], highs[row - 1])
        self.assertLess(highs[-1], self.ring.num_tokens(6))

    def test_extend_drops_rows_older_than_one_ring(self):
        """One prompt pass writes every row at once: rows older than one ring
        must go to the trash, so the kept rows are distinct ring slots."""
        prefix, extend = [10, 100], [20, 3000]
        positions = torch.cat([torch.arange(10, 30), torch.arange(100, 3100)])
        slots = self.ring.extend_slots(
            req_rows=torch.tensor([3, 4]),
            prefix_lens=prefix,
            extend_lens=extend,
            positions=positions,
        )
        self.assertEqual(
            slots[:20].tolist(),
            self.ring.slots(torch.tensor(3), positions[:20]).tolist(),
        )
        long_slots, long_pos = slots[20:], positions[20:]
        kept = long_pos >= 3100 - self.ring.ring_tokens
        self.assertEqual(long_slots[kept].unique().numel(), self.ring.ring_tokens)
        self.assertTrue(bool((long_slots[~kept] < self.ring.trash_tokens).all()))


if __name__ == "__main__":
    unittest.main()
