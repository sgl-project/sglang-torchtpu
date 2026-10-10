"""Unit tests for chunked-prefill pull pairing.

Pure index math, so these run anywhere -- no TPU / torch_tpu / tpu_raiden
required.

Run: python -m pytest test/manual/kv_transfer/test_raiden_chunk_plan.py
"""

import unittest

from sglang_tpu.disaggregation.raiden.chunk_plan import (
    chunk_pull_window,
    whole_pull_registration,
)


def _pair(chunks, total, local_pages):
    """Replays chunked announcements and returns the (remote, local) pairing."""
    remote_ids = list(range(1000, 1000 + total))
    local_ids = [2000 + i for i in range(local_pages)]
    paired_remote, paired_local = [], []
    start = 0
    for size in chunks:
        window = chunk_pull_window(start, size, total, local_pages)
        if window:
            chunk_ids = remote_ids[start : start + size]
            paired_remote += chunk_ids[
                window.chunk_offset : window.chunk_offset + window.length
            ]
            paired_local += local_ids[
                window.local_offset : window.local_offset + window.length
            ]
        start += size
    return paired_remote, paired_local


def _whole(total, local_pages):
    """The pairing the single-shot path produces, for equivalence checks."""
    remote_ids = list(range(1000, 1000 + total))
    local_ids = [2000 + i for i in range(local_pages)]
    if not local_ids:
        return [], []
    if len(remote_ids) > len(local_ids):
        remote_ids = remote_ids[-len(local_ids) :]
    elif len(remote_ids) < len(local_ids):
        local_ids = local_ids[: len(remote_ids)]
    return remote_ids, local_ids


class TestChunkPullWindow(unittest.TestCase):
    def test_aligned_chunk_is_taken_whole(self):
        self.assertEqual(chunk_pull_window(0, 16, 64, 64), (0, 0, 16))
        self.assertEqual(chunk_pull_window(16, 16, 64, 64), (0, 16, 16))

    def test_pages_the_decode_already_caches_are_dropped(self):
        # Decode holds only the last 40 of 64 prefill pages.
        self.assertFalse(chunk_pull_window(0, 16, 64, 40))
        self.assertEqual(chunk_pull_window(16, 16, 64, 40), (8, 0, 8))
        self.assertEqual(chunk_pull_window(32, 16, 64, 40), (0, 8, 16))

    def test_reserved_decode_page_is_past_the_end(self):
        # Decode allocated one page more than the prefill fills.
        self.assertEqual(chunk_pull_window(48, 16, 64, 65), (0, 48, 16))
        self.assertFalse(chunk_pull_window(64, 1, 64, 65))

    def test_full_prefix_hit_pulls_nothing(self):
        self.assertFalse(chunk_pull_window(0, 16, 64, 0))

    def test_degenerate_inputs(self):
        self.assertFalse(chunk_pull_window(0, 0, 64, 64))
        self.assertFalse(chunk_pull_window(0, 16, 0, 64))
        self.assertFalse(chunk_pull_window(-1, 16, 64, 64))


class TestEquivalenceWithSingleShot(unittest.TestCase):
    def _check(self, chunks, total, local_pages):
        self.assertEqual(_pair(chunks, total, local_pages), _whole(total, local_pages))

    def test_matches_when_chunked(self):
        self._check([16, 16, 16, 16], 64, 64)

    def test_matches_with_a_ragged_last_chunk(self):
        self._check([16, 16, 16, 5], 53, 53)

    def test_matches_with_a_decode_prefix_hit(self):
        self._check([16, 16, 16, 16], 64, 40)

    def test_matches_with_a_reserved_decode_page(self):
        self._check([16, 16, 16, 16], 64, 65)

    def test_matches_for_a_single_chunk(self):
        self._check([64], 64, 64)
        self._check([64], 64, 40)


class TestWholePullRegistration(unittest.TestCase):
    """Verifies that producer registration matches the tail pages read by decode."""

    BLOCKS = list(range(1000, 1064))

    def test_no_hit_registers_everything(self):
        self.assertEqual(whole_pull_registration(self.BLOCKS, 64), self.BLOCKS)

    def test_reserved_decode_page_registers_everything(self):
        self.assertEqual(whole_pull_registration(self.BLOCKS, 65), self.BLOCKS)

    def test_decode_prefix_hit_registers_the_tail(self):
        self.assertEqual(whole_pull_registration(self.BLOCKS, 40), self.BLOCKS[-40:])

    def test_full_prefix_hit_registers_nothing(self):
        self.assertEqual(whole_pull_registration(self.BLOCKS, 0), [])
        self.assertEqual(whole_pull_registration([], 0), [])

    def test_matches_the_single_shot_pairing(self):
        for local_pages in (0, 1, 40, 63, 64, 65):
            remote, _ = _whole(len(self.BLOCKS), local_pages)
            self.assertEqual(
                whole_pull_registration(self.BLOCKS, local_pages),
                remote,
                f"local_pages={local_pages}",
            )


if __name__ == "__main__":
    unittest.main()
