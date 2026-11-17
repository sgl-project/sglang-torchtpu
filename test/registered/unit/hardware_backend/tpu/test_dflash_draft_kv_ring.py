"""DraftKvRing slot math and the draft KV writer's spans over it: the ring the
DFlash draft reads must hold what it wrote."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.dflash_draft_kv_write import _extend_spans
from sglang_tpu.experimental.speculative.dflash.draft_kv_ring import DraftKvRing

_WINDOW, _WIDTH, _PAGE = 1024, 8, 128


def _write(cache_loc: torch.Tensor, pool_tokens: int, max_num_reqs: int):
    """Replay _extend_spans' spans as the kernel's DMAs: pool slot -> source row."""
    dst, src, lens, max_len = _extend_spans(
        page_size=_PAGE, cache_loc=cache_loc, max_num_reqs=max_num_reqs
    )
    pool = torch.full((pool_tokens,), -1, dtype=torch.int64)
    for d, s, n in zip(dst.tolist(), src.tolist(), lens.tolist()):
        assert n <= max_len and (d % _PAGE) + n <= _PAGE, (d, n)
        pool[d : d + n] = torch.arange(s, s + n)
    return pool, int(lens.sum())


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
        self.assertTrue(bool((long_slots[~kept] == 0).all()))

    def test_writer_puts_every_kept_row_in_its_own_ring(self):
        """Through the writer's spans: every kept position holds its own row,
        dropped rows (slot 0, anywhere in the batch) are not written, and no
        write lands in another request's ring. Covers a first request longer
        than one ring, a cached prefix, mixed lengths and tail padding."""
        rows = 6
        pool_tokens = self.ring.num_tokens(rows)
        for reqs, pad in (
            ([(1, 0, 4096)], 0),
            ([(2, 512, 2048)], 0),
            ([(0, 0, 300), (1, 0, 3000), (4, 7000, 129)], 0),
            ([(3, 0, 1500), (5, 1281, 1)], 64),
        ):
            with self.subTest(reqs=reqs, pad=pad):
                positions = torch.cat([torch.arange(p, p + n) for _, p, n in reqs])
                slots = self.ring.extend_slots(
                    req_rows=torch.tensor([r for r, _, _ in reqs]),
                    prefix_lens=[p for _, p, _ in reqs],
                    extend_lens=[n for _, _, n in reqs],
                    positions=positions,
                )
                cache_loc = torch.cat([slots, torch.zeros(pad, dtype=slots.dtype)])
                pool, written = _write(cache_loc, pool_tokens, len(reqs) + 1)
                self.assertEqual(written, int((slots != 0).sum()))
                self.assertTrue(bool((pool[: self.ring.trash_tokens] == -1).all()))
                offset = 0
                for row, prefix, n in reqs:
                    end = prefix + n
                    for pos in range(max(prefix, end - self.ring.ring_tokens), end):
                        slot = int(
                            self.ring.slots(torch.tensor(row), torch.tensor(pos))
                        )
                        self.assertEqual(int(pool[slot]), offset + pos - prefix)
                    offset += n
                owner = (
                    torch.arange(pool_tokens) - self.ring.trash_tokens
                ) // self.ring.ring_tokens
                src_row = torch.cat(
                    [torch.full((n,), r) for r, _, n in reqs] + [torch.full((pad,), -1)]
                )
                hit = pool >= 0
                self.assertTrue(bool((src_row[pool[hit]] == owner[hit]).all()))

    def test_writer_spans_cover_allocator_slots(self):
        """Without the ring: requests resuming mid-page, then tail padding;
        every token lands on its own slot."""
        loc = torch.cat(
            [
                torch.arange(page * _PAGE + off, page * _PAGE + off + n)
                for page, off, n in ((3, 5, 300), (9, 0, 1), (12, 127, 200))
            ]
            + [torch.zeros(37, dtype=torch.int64)]
        )
        pool, written = _write(loc, 20 * _PAGE, 3)
        real = loc != 0
        self.assertEqual(written, int(real.sum()))
        self.assertEqual(
            pool[loc[real]].tolist(), torch.nonzero(real).flatten().tolist()
        )


if __name__ == "__main__":
    unittest.main()
