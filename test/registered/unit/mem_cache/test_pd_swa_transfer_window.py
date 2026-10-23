"""Edge cases of the P/D sliding-window transfer window (#1391).

``pd_swa_transfer_window_start`` decides which SWA pages the prefill sends,
the decode allocates and the KV checksum covers. These tests sweep window
sizes, page sizes and prompt lengths well past the handful of combinations the
issue reproduced, and tie the decode gate, the checksum range and the Raiden
staging limit to the same helper.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.disaggregation.checksum import state_indices_for_request
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.mem_cache.common import (
    page_align_floor,
    pd_swa_transfer_max_pages,
    pd_swa_transfer_window_start,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase, enter_override

PAGE_SIZES = (1, 16, 64, 256, 512)


def window_sizes(page):
    """Windows worth checking at this page size: below, at and just past a
    page, odd residues, multiples, and the deployed shapes (gpt-oss 127/64,
    2047/512, the unit tests' 1000/512, 4096)."""
    windows = {1, page - 1, page, page + 1, 2 * page - 1, 3 * page}
    windows |= {127, 1000, 2047, 4096}
    return sorted(w for w in windows if w >= 1)


def prompt_lengths(window, page):
    """Prompt lengths around every interesting boundary: shorter than a window,
    the window itself and one past it, every residue of the first pages past
    the window, and a few huge prompts at each residue."""
    lengths = set(range(1, min(window, 3 * page) + 1))
    lengths |= {window, window + 1}
    lengths |= set(range(max(1, window - page), window + 4 * page + 1))
    for base in (1 << 20, (1 << 20) + 7 * page):
        for residue in {0, 1, page - 1, window % page, (window % page) - 1}:
            if 0 <= residue < page:
                lengths.add(base + residue)
    return sorted(lengths)


def old_window_start(seq_len, window, page):
    """The pre-#1391 window start, ``page_floor(max(0, L - W))``."""
    return page_align_floor(max(0, seq_len - window), page)


def pages_from(start, seq_len, page):
    """Pages a window starting at ``start`` occupies up to ``seq_len``."""
    return -(-seq_len // page) - start // page


def gate_passes(start, seq_len, window, page, eagle=False):
    """``_can_ignore_swa_kv_miss``'s re-match gate for a tail starting at
    ``start``: a full window must sit before the page-aligned insert boundary
    of the cache key, which is one token shorter under EAGLE."""
    key_len = seq_len - 1 if eagle else seq_len
    return page_align_floor(key_len, page) - start >= window


class TestTransferWindowStartProperties(unittest.TestCase):
    def _check(self, assertion):
        failures = []
        for page in PAGE_SIZES:
            for window in window_sizes(page):
                for seq_len in prompt_lengths(window, page):
                    msg = assertion(seq_len, window, page)
                    if msg:
                        failures.append(f"W={window} P={page} L={seq_len}: {msg}")
        self.assertEqual(failures[:20], [], f"{len(failures)} failures")

    def test_start_is_page_aligned_and_within_bounds(self):
        def check(seq_len, window, page):
            start = pd_swa_transfer_window_start(seq_len, window, page)
            old = old_window_start(seq_len, window, page)
            if start % page:
                return f"start {start} not page aligned"
            if not 0 <= start <= old:
                return f"start {start} outside [0, {old}]"
            if start < old - page:
                return f"start {start} more than a page before {old}"
            if start >= seq_len and seq_len > 0 and start > 0:
                return f"start {start} at or past the prompt end"

        self._check(check)

    def test_tail_holds_a_full_window_before_the_insert_boundary(self):
        def check(seq_len, window, page):
            start = pd_swa_transfer_window_start(seq_len, window, page)
            boundary = page_align_floor(seq_len, page)
            if boundary >= window and boundary - start < window:
                return f"only {boundary - start} tokens before boundary {boundary}"
            if boundary < window and start != 0:
                return f"prompt shorter than a window page-aligned, start {start}"

        self._check(check)

    def test_window_never_exceeds_the_staging_limit(self):
        def check(seq_len, window, page):
            start = pd_swa_transfer_window_start(seq_len, window, page)
            pages = pages_from(start, seq_len, page)
            limit = pd_swa_transfer_max_pages(window, page)
            if pages > limit:
                return f"{pages} pages exceed the limit {limit}"
            old_pages = pages_from(
                old_window_start(seq_len, window, page), seq_len, page
            )
            if pages - old_pages not in (0, 1):
                return f"{pages} pages vs {old_pages} with the old start"

        self._check(check)

    def test_limit_is_reached_for_some_length_at_every_shape(self):
        # ceil(W / P) + 1 is tight, not just an upper bound: some prompt
        # length needs exactly that many pages whenever W % P > 0.
        for page in PAGE_SIZES:
            for window in window_sizes(page):
                if window % page == 0:
                    continue
                limit = pd_swa_transfer_max_pages(window, page)
                needed = max(
                    pages_from(
                        pd_swa_transfer_window_start(seq_len, window, page),
                        seq_len,
                        page,
                    )
                    for seq_len in prompt_lengths(window, page)
                )
                self.assertEqual(needed, limit, f"W={window} P={page}")

    def test_moves_one_page_earlier_exactly_where_the_old_start_failed_the_gate(self):
        def check(seq_len, window, page):
            start = pd_swa_transfer_window_start(seq_len, window, page)
            old = old_window_start(seq_len, window, page)
            moved = start != old
            if moved and start != old - page:
                return f"moved by {old - start}, not one page"
            formula = window % page > 0 and seq_len % page >= window % page
            if old > 0 and moved != formula:
                return f"moved={moved} but W%P>0 and L%P>=W%P is {formula}"
            if seq_len - window >= page:
                old_rejected = not gate_passes(old, seq_len, window, page)
                if moved != old_rejected:
                    return f"moved={moved} but the old start rejected={old_rejected}"
            elif moved:
                return "moved although the old start was already page 0"

        self._check(check)

    def test_page_multiple_windows_and_page_size_one_never_move(self):
        for page in PAGE_SIZES:
            for window in (page, 2 * page, 3 * page, 8 * page):
                for seq_len in prompt_lengths(window, page):
                    self.assertEqual(
                        pd_swa_transfer_window_start(seq_len, window, page),
                        old_window_start(seq_len, window, page),
                        f"W={window} P={page} L={seq_len}",
                    )
                self.assertEqual(
                    pd_swa_transfer_max_pages(window, page), window // page + 1
                )
        for window in (1, 2, 127, 4096):
            for seq_len in prompt_lengths(window, 1):
                self.assertEqual(
                    pd_swa_transfer_window_start(seq_len, window, 1),
                    max(0, seq_len - window),
                )
            self.assertEqual(pd_swa_transfer_max_pages(window, 1), window + 1)

    def test_degenerate_inputs(self):
        self.assertEqual(pd_swa_transfer_window_start(0, 127, 64), 0)
        self.assertEqual(pd_swa_transfer_window_start(-5, 127, 64), 0)
        for seq_len in range(0, 300):
            # A window longer than the prompt sends everything.
            self.assertEqual(pd_swa_transfer_window_start(seq_len, 300, 64), 0)
            # A one-token window keeps the page before the boundary.
            self.assertEqual(
                pd_swa_transfer_window_start(seq_len, 1, 64),
                max(0, page_align_floor(seq_len, 64) - 64),
            )
        self.assertEqual(pd_swa_transfer_max_pages(1, 64), 2)
        self.assertEqual(pd_swa_transfer_max_pages(64, 64), 2)
        self.assertEqual(pd_swa_transfer_max_pages(65, 64), 3)
        self.assertEqual(pd_swa_transfer_max_pages(127, 64), 3)
        self.assertEqual(pd_swa_transfer_max_pages(128, 64), 3)
        self.assertEqual(pd_swa_transfer_max_pages(2047, 512), 5)
        self.assertEqual(pd_swa_transfer_max_pages(4096, 128), 33)


def _swa_tail_queue(window, page, *, eagle=False, hicache=False):
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
    queue.scheduler = SimpleNamespace(sliding_window_size=window)
    queue.token_to_kv_pool_allocator = MagicMock(page_size=page)
    queue.tree_cache = SimpleNamespace(
        tree_core=SimpleNamespace(is_eagle=eagle, enable_hicache=hicache)
    )
    return queue


def _req(fill_len):
    return SimpleNamespace(origin_input_ids=[0] * fill_len, output_ids=[])


class TestDecodeTailAndGate(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="scheduler")
        enter_override(
            self,
            get_context().override_server_args(
                disaggregation_decode_enable_radix_cache=True
            ),
        )

    def test_tail_starts_where_the_prefill_window_starts_for_every_shape(self):
        failures = []
        for page in PAGE_SIZES[1:]:  # tail prealloc needs page_size > 1
            for window in window_sizes(page):
                queue = _swa_tail_queue(window, page)
                for seq_len in prompt_lengths(window, page):
                    tail = queue._swa_tail_len(seq_len)
                    start = pd_swa_transfer_window_start(seq_len, window, page)
                    if seq_len - tail != start or tail <= 0:
                        failures.append((window, page, seq_len, tail, start))
        self.assertEqual(failures[:20], [], f"{len(failures)} failures")

    def test_tail_is_the_whole_prompt_without_tail_prealloc_or_window(self):
        queue = _swa_tail_queue(127, 64)
        queue._uses_swa_tail_prealloc.return_value = False
        self.assertEqual(queue._swa_tail_len(1023), 1023)
        self.assertEqual(queue._swa_tail_len(0), 0)
        self.assertEqual(queue._swa_tail_len(-3), 0)
        queue = _swa_tail_queue(None, 64)
        self.assertEqual(queue._swa_tail_len(1023), 1023)
        queue = _swa_tail_queue(0, 64)
        self.assertEqual(queue._swa_tail_len(1023), 1023)

    def test_gate_passes_every_length_once_a_window_fits_before_the_boundary(self):
        failures = []
        for page in PAGE_SIZES[1:]:
            for window in window_sizes(page):
                queue = _swa_tail_queue(window, page)
                for seq_len in prompt_lengths(window, page):
                    passes = queue._can_ignore_swa_kv_miss(_req(seq_len))
                    # Shorter prompts hold no full window before the boundary
                    # whatever the start, so nothing before the tail is reusable.
                    expected = page_align_floor(seq_len, page) >= window
                    if passes != expected:
                        failures.append((window, page, seq_len, passes))
        self.assertEqual(failures[:20], [], f"{len(failures)} failures")

    def test_gate_with_eagle_fails_only_page_aligned_prompts(self):
        # EAGLE's bigram key is one token shorter, so for L % P == 0 the insert
        # boundary is a page lower than the one the window start is derived
        # from. Those lengths are the known remaining gap of #1391.
        failures = []
        for page in PAGE_SIZES[1:]:
            for window in window_sizes(page):
                queue = _swa_tail_queue(window, page, eagle=True)
                for seq_len in prompt_lengths(window, page):
                    if page_align_floor(seq_len - 1, page) < window:
                        continue
                    passes = queue._can_ignore_swa_kv_miss(_req(seq_len))
                    expected = seq_len % page != 0
                    if passes != expected:
                        failures.append((window, page, seq_len, passes, expected))
        self.assertEqual(failures[:20], [], f"{len(failures)} failures")

    def test_gate_is_closed_by_hicache_and_without_tail_prealloc(self):
        for seq_len in (1023, 1024, 1087, 4095, 4096):
            self.assertFalse(
                _swa_tail_queue(127, 64, hicache=True)._can_ignore_swa_kv_miss(
                    _req(seq_len)
                )
            )
            queue = _swa_tail_queue(127, 64)
            queue._uses_swa_tail_prealloc.return_value = False
            self.assertFalse(queue._can_ignore_swa_kv_miss(_req(seq_len)))

    def test_old_start_failed_the_gate_at_the_reported_lengths(self):
        # The lengths #1391 measured: W=2047/P=512 at L % 512 == 511 and
        # gpt-oss W=127/P=64 at L % 64 == 63 lost reuse; the neighbours did not.
        for window, page, bad_residue in ((2047, 512, 511), (127, 64, 63)):
            queue = _swa_tail_queue(window, page)
            for seq_len in range(4 * page + window, 8 * page + window):
                old = old_window_start(seq_len, window, page)
                self.assertEqual(
                    not gate_passes(old, seq_len, window, page),
                    seq_len % page == bad_residue,
                    f"W={window} P={page} L={seq_len}",
                )
                self.assertTrue(queue._can_ignore_swa_kv_miss(_req(seq_len)))


class TestChecksumRangeMatchesTheTransfer(unittest.TestCase):
    ROWS = 1 << 15

    def _scheduler(self, window, page):
        pool = SWAKVPool.__new__(SWAKVPool)
        return SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=lambda: pool,
                page_size=page,
                translate_loc_from_full_to_swa=lambda t: t,
            ),
            sliding_window_size=window,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(self.ROWS, dtype=torch.int64).view(
                    1, self.ROWS
                )
            ),
        )

    def test_checksum_pages_equal_the_decode_tail_and_the_prefill_window(self):
        req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=0))
        failures = []
        for page in PAGE_SIZES[1:]:
            for window in window_sizes(page):
                sched = self._scheduler(window, page)
                queue = _swa_tail_queue(window, page)
                for seq_len in prompt_lengths(window, page):
                    if seq_len > self.ROWS:
                        continue
                    start = pd_swa_transfer_window_start(seq_len, window, page)
                    checksum_pages = state_indices_for_request(
                        sched, req, seq_len
                    ).tolist()
                    # The prefill payload and the decode tail both page the
                    # same [start, seq_len) rows of req_to_token.
                    prefill_pages = list(range(start // page, -(-seq_len // page)))
                    tail_start = seq_len - queue._swa_tail_len(seq_len)
                    decode_pages = list(range(tail_start // page, -(-seq_len // page)))
                    if not checksum_pages == prefill_pages == decode_pages:
                        failures.append(
                            (
                                window,
                                page,
                                seq_len,
                                checksum_pages,
                                prefill_pages,
                                decode_pages,
                            )
                        )
                    if len(prefill_pages) > pd_swa_transfer_max_pages(window, page):
                        failures.append((window, page, seq_len, "over limit"))
        self.assertEqual(failures[:10], [], f"{len(failures)} failures")

    def test_version_skew_changes_the_page_count_exactly_at_the_moved_lengths(self):
        # An old-formula prefill announces one page less than a new decode
        # allocates exactly for the lengths the fix moves; the Raiden guard
        # keys on that difference (see test_tpu_raiden_offload_unit).
        for page in PAGE_SIZES[1:]:
            for window in window_sizes(page):
                for seq_len in prompt_lengths(window, page):
                    new = pd_swa_transfer_window_start(seq_len, window, page)
                    old = old_window_start(seq_len, window, page)
                    diff = pages_from(new, seq_len, page) - pages_from(
                        old, seq_len, page
                    )
                    self.assertEqual(
                        diff, 1 if new != old else 0, f"W={window} P={page} L={seq_len}"
                    )


if __name__ == "__main__":
    unittest.main()
