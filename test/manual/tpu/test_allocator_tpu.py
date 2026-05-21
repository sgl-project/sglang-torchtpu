"""Unit tests for :class:`TpuPagedTokenToKVPoolAllocator`."""

import unittest

import torch

from sglang.test.test_utils import CustomTestCase


def _make_allocator(
    size: int = 32, page_size: int = 4, *, device: str = "cpu", need_sort: bool = False
):
    """Build an allocator. ``size`` must be a multiple of ``page_size``.
    ``kvcache=None`` is fine — the base class only stores the reference and
    never calls into it from the allocator path. ``device`` defaults to CPU
    (host-only logic); the hardware-gated tests pass ``device="tpu"``."""
    from sglang.srt.hardware_backend.tpu.allocator_tpu import (
        TpuPagedTokenToKVPoolAllocator,
    )

    return TpuPagedTokenToKVPoolAllocator(
        size=size,
        page_size=page_size,
        dtype=torch.bfloat16,
        device=device,
        kvcache=None,
        need_sort=need_sort,
    )


def _detect_tpu() -> bool:
    """True only when torch_tpu is installed AND a ``tpu`` device is usable.
    The ``import torch_tpu`` (which registers the PrivateUse1 backend and
    initialises PJRT) is gated behind ``find_spec`` so CPU-only machines incur
    no side effects at collection time."""
    import importlib.util

    if importlib.util.find_spec("torch_tpu") is None:
        return False
    try:
        import torch_tpu  # noqa: F401

        torch.zeros(1, dtype=torch.int32).to("tpu")
        return True
    except Exception:
        return False


_TPU_AVAILABLE = _detect_tpu()


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _expected_new_pages(prefix_lens, seq_lens, page_size: int) -> int:
    """Pages an extend should consume = sum of per-request page deltas."""
    return sum(
        _ceil_div(int(s), page_size) - _ceil_div(int(p), page_size)
        for p, s in zip(prefix_lens, seq_lens)
    )


class TestAllocatorPoolLifecycle(CustomTestCase):
    """Pool initialisation, accounting, and clear()."""

    def test_pool_starts_full_with_slot_zero_reserved(self):
        a = _make_allocator(size=32, page_size=4)
        # num_pages = 8; slot 0 reserved for the dummy-padding page;
        # so 8 usable pages with IDs 1..8.
        self.assertEqual(a.num_pages, 8)
        self.assertEqual(len(a.free_pages), 8)
        self.assertEqual(int(a.free_pages.min()), 1)
        self.assertEqual(int(a.free_pages.max()), 8)
        # available_size reports usable tokens, not pages.
        self.assertEqual(a.available_size(), 8 * 4)

    def test_clear_restores_full_pool(self):
        a = _make_allocator(size=32, page_size=4)
        # Drain a couple of pages, then clear() must restore IDs 1..8.
        out = a.alloc(8)  # 2 pages worth of tokens
        self.assertIsNotNone(out)
        self.assertEqual(len(a.free_pages), 6)
        a.clear()
        self.assertEqual(len(a.free_pages), 8)
        self.assertEqual(int(a.free_pages.min()), 1)


class TestAllocDecode(CustomTestCase):
    """``alloc_decode`` is the hot path the rewrite optimises. Output
    indices follow the device formula::

        need_new_pages[i] == 0  ->  out_indices[i] = last_loc[i] + 1
        need_new_pages[i] == 1  ->  out_indices[i] = new_page_id * page_size
    """

    def test_no_new_pages_continues_current_page(self):
        a = _make_allocator(size=32, page_size=4)
        # Two requests, both mid-page (seq_lens % 4 != 1). The allocator
        # should NOT consume any free pages — out_indices = last_loc + 1.
        seq_lens = torch.tensor([3, 7], dtype=torch.int32)
        last_loc = torch.tensor([5, 11], dtype=torch.int32)
        free_before = len(a.free_pages)

        out = a.alloc_decode(seq_lens, seq_lens.clone(), last_loc)

        self.assertTrue(torch.equal(out, torch.tensor([6, 12], dtype=torch.int32)))
        self.assertEqual(len(a.free_pages), free_before)

    def test_all_new_pages_assigns_fresh_page_per_req(self):
        a = _make_allocator(size=32, page_size=4)
        # Both reqs at a page boundary (seq_lens % 4 == 1).
        seq_lens = torch.tensor([5, 9], dtype=torch.int32)
        # Both cross a boundary -> last_loc lands on a full page's last slot
        # (15, 23); unused once a fresh page is assigned, but kept valid/in-range.
        last_loc = torch.tensor([15, 23], dtype=torch.int32)
        first_three_free = list(map(int, a.free_pages[:3]))

        out = a.alloc_decode(seq_lens, seq_lens.clone(), last_loc)

        # Each req gets the NEXT distinct page (in order). Pool was
        # [1, 2, 3, ...] so req 0 -> page 1, req 1 -> page 2.
        expected = torch.tensor(
            [first_three_free[0] * 4, first_three_free[1] * 4],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(out, expected))
        # Consumed exactly two pages; rest stay.
        self.assertEqual(len(a.free_pages), 8 - 2)
        self.assertEqual(int(a.free_pages[0]), first_three_free[2])

    def test_mixed_decode_some_cross_boundary(self):
        a = _make_allocator(size=32, page_size=4)
        # Req 0 continues mid-page; req 1 crosses into a fresh page. Both
        # last_loc values are valid in-range slots consistent with the decode
        # invariant ((last_loc + 2) % 4 == seq_len % 4): req 0 at slot 9
        # (mid page 2), req 1 at slot 15 (its prior page is full, so the next
        # token needs a new page — though last_loc is then unused on that lane).
        seq_lens = torch.tensor([3, 5], dtype=torch.int32)
        last_loc = torch.tensor([9, 15], dtype=torch.int32)
        first_free = int(a.free_pages[0])

        out = a.alloc_decode(seq_lens, seq_lens.clone(), last_loc)

        self.assertEqual(int(out[0]), 10)  # last_loc[0] + 1, no new page
        self.assertEqual(int(out[1]), first_free * 4)  # new page * page_size
        # Exactly one page consumed.
        self.assertEqual(len(a.free_pages), 7)

    def test_returns_none_when_pool_exhausted(self):
        a = _make_allocator(size=8, page_size=4)  # only 2 usable pages
        # Drain both. last_loc=7 is the last slot of page 1 (valid for size=8).
        a.alloc_decode(
            torch.tensor([5, 5], dtype=torch.int32),
            torch.tensor([5, 5], dtype=torch.int32),
            torch.tensor([7, 7], dtype=torch.int32),
        )
        # Now ask for one more page — should return None, not crash.
        out = a.alloc_decode(
            torch.tensor([5], dtype=torch.int32),
            torch.tensor([5], dtype=torch.int32),
            torch.tensor([7], dtype=torch.int32),
        )
        self.assertIsNone(out)

    def test_decode_single_request_continue(self):
        a = _make_allocator(size=32, page_size=4)
        # bs=1, mid-page (6 % 4 == 2): no page consumed, out = last_loc + 1.
        out = a.alloc_decode(
            torch.tensor([6], dtype=torch.int32),
            torch.tensor([6], dtype=torch.int32),
            torch.tensor([10], dtype=torch.int32),
        )
        self.assertEqual(out.shape, (1,))
        self.assertEqual(int(out[0]), 11)
        self.assertEqual(len(a.free_pages), 8)

    def test_decode_single_request_new_page(self):
        a = _make_allocator(size=32, page_size=4)
        first_free = int(a.free_pages[0])
        # bs=1, at boundary (5 % 4 == 1): one fresh page. last_loc=15 is a
        # full page's last slot (valid/in-range); unused once a page is assigned.
        out = a.alloc_decode(
            torch.tensor([5], dtype=torch.int32),
            torch.tensor([5], dtype=torch.int32),
            torch.tensor([15], dtype=torch.int32),
        )
        self.assertEqual(out.shape, (1,))
        self.assertEqual(int(out[0]), first_free * 4)
        self.assertEqual(len(a.free_pages), 7)

    def test_decode_large_batch_all_continue(self):
        a = _make_allocator(size=64, page_size=4)
        bs = 8
        # No seq_len hits a page boundary (none % 4 == 1) -> all continue.
        seq_lens = torch.tensor([2, 3, 4, 6, 7, 8, 10, 11], dtype=torch.int32)
        last_loc = torch.arange(1, bs + 1, dtype=torch.int32)
        free_before = len(a.free_pages)

        out = a.alloc_decode(seq_lens, seq_lens.clone(), last_loc)

        self.assertEqual(out.shape, (bs,))
        self.assertTrue(torch.equal(out, last_loc + 1))
        self.assertEqual(len(a.free_pages), free_before)  # no pages consumed

    def test_decode_large_batch_mixed_assigns_pages_in_order(self):
        a = _make_allocator(size=64, page_size=4)
        # bs=6, pattern: continue, new, continue, new, continue, new.
        # last_loc values are valid in-range slots satisfying the decode
        # invariant ((last_loc + 2) % 4 == seq_len % 4); the "new" lanes sit
        # on a full page's last slot.
        seq_lens = torch.tensor([3, 5, 7, 9, 4, 13], dtype=torch.int32)
        last_loc = torch.tensor([21, 15, 29, 23, 38, 31], dtype=torch.int32)
        f = list(map(int, a.free_pages[:3]))  # first three free page IDs

        out = a.alloc_decode(seq_lens, seq_lens.clone(), last_loc)

        # Continues take last_loc+1; new pages take free pages in order.
        expected = torch.tensor(
            [22, f[0] * 4, 30, f[1] * 4, 39, f[2] * 4], dtype=torch.int32
        )
        self.assertTrue(torch.equal(out, expected))
        self.assertEqual(len(a.free_pages), 16 - 3)

    def test_decode_nondefault_page_size_eight(self):
        a = _make_allocator(size=64, page_size=8)
        first_free = int(a.free_pages[0])
        # req0 continues (8 % 8 == 0), req1 crosses a boundary (9 % 8 == 1).
        # last_loc are valid in-range slots; req1 sits on a full page's end.
        seq_lens = torch.tensor([8, 9], dtype=torch.int32)
        last_loc = torch.tensor([15, 23], dtype=torch.int32)

        out = a.alloc_decode(seq_lens, seq_lens.clone(), last_loc)

        self.assertTrue(
            torch.equal(out, torch.tensor([16, first_free * 8], dtype=torch.int32))
        )
        self.assertEqual(len(a.free_pages), 8 - 1)

    def test_decode_varied_batch_sizes_preserve_invariants(self):
        # Sweep batch sizes / boundary patterns; assert the invariants that
        # hold for ANY valid decode step. last_loc values are distinct,
        # in-range slots (100 + i) chosen high enough that "continue" outputs
        # never collide with the freshly-assigned low pages.
        configs = [
            (4, [5]),
            (4, [4, 6]),
            (4, [5, 9, 13]),
            (4, [3, 5, 7, 9, 4, 13, 8, 17]),
            (8, [8, 9, 16, 17]),
        ]
        for page_size, seq in configs:
            with self.subTest(page_size=page_size, seq=seq):
                a = _make_allocator(size=256, page_size=page_size)
                bs = len(seq)
                seq_t = torch.tensor(seq, dtype=torch.int32)
                last_loc = torch.tensor([100 + i for i in range(bs)], dtype=torch.int32)
                expected_pages = sum(1 for s in seq if s % page_size == 1)
                free_before = len(a.free_pages)

                out = a.alloc_decode(seq_t, seq_t.clone(), last_loc)

                self.assertIsNotNone(out)
                self.assertEqual(out.shape, (bs,))
                self.assertEqual(out.dtype, torch.int32)
                self.assertEqual(len(a.free_pages), free_before - expected_pages)
                # Each request either continues (last_loc+1) or lands on a
                # fresh, page-aligned slot.
                for i, s in enumerate(seq):
                    if s % page_size == 1:
                        self.assertEqual(int(out[i]) % page_size, 0)
                    else:
                        self.assertEqual(int(out[i]), 100 + i + 1)


class TestFreeRoundTrip(CustomTestCase):
    """``free`` accepts a device tensor of token indices and returns
    the corresponding page IDs to the pool."""

    def test_free_returns_pages_to_pool(self):
        a = _make_allocator(size=32, page_size=4)
        # Use alloc() to drain two pages (8 contiguous tokens).
        out = a.alloc(8)
        self.assertIsNotNone(out)
        self.assertEqual(len(a.free_pages), 6)

        # Free those 8 tokens; the two pages they came from should
        # re-enter the pool.
        a.free(out)
        self.assertEqual(len(a.free_pages), 8)

    def test_free_dedupes_same_page(self):
        a = _make_allocator(size=32, page_size=4)
        # All tokens belong to page 1 — freeing them should return
        # exactly one page to the pool.
        tok = torch.tensor([4, 5, 6, 7], dtype=torch.int64)
        before = len(a.free_pages)
        a.free(tok)
        self.assertEqual(len(a.free_pages), before + 1)

    def test_free_empty_is_noop(self):
        a = _make_allocator(size=32, page_size=4)
        before = len(a.free_pages)
        a.free(torch.empty(0, dtype=torch.int64))
        self.assertEqual(len(a.free_pages), before)


class TestAllocExtend(CustomTestCase):
    """``alloc_extend`` is the prefill / extend path. Less optimisation-
    critical than ``alloc_decode`` but still has to compute the right
    number of new pages from host tensors."""

    def test_extend_consumes_expected_pages(self):
        a = _make_allocator(size=32, page_size=4)
        # One request, prefix=0 tokens, target seq_lens=5 -> two pages
        # (one full + one partial); extend_num_tokens=5.
        prefix_lens = torch.tensor([0], dtype=torch.int32)
        seq_lens = torch.tensor([5], dtype=torch.int32)
        last_loc = torch.tensor([-1], dtype=torch.int32)

        out = a.alloc_extend(
            prefix_lens,
            prefix_lens.clone(),
            seq_lens,
            seq_lens.clone(),
            last_loc,
            extend_num_tokens=5,
        )
        # extend_num_tokens output slots.
        self.assertEqual(out.shape, (5,))
        # Two pages consumed (the formula in the kernel is roundup-based).
        self.assertEqual(len(a.free_pages), 8 - 2)

    def test_extend_returns_none_when_pool_too_small(self):
        a = _make_allocator(size=8, page_size=4)  # 2 usable pages
        prefix_lens = torch.tensor([0], dtype=torch.int32)
        seq_lens = torch.tensor([100], dtype=torch.int32)  # >> 2 pages
        last_loc = torch.tensor([-1], dtype=torch.int32)
        out = a.alloc_extend(
            prefix_lens,
            prefix_lens.clone(),
            seq_lens,
            seq_lens.clone(),
            last_loc,
            extend_num_tokens=100,
        )
        self.assertIsNone(out)

    def test_extend_multiple_requests(self):
        a = _make_allocator(size=32, page_size=4)
        # bs=3, all from scratch (prefix=0): pages = 1 + 2 + 2 = 5.
        prefix_lens = torch.tensor([0, 0, 0], dtype=torch.int32)
        seq_lens = torch.tensor([4, 8, 5], dtype=torch.int32)
        last_loc = torch.tensor([-1, -1, -1], dtype=torch.int32)
        extend = int((seq_lens - prefix_lens).sum())  # 17
        pages = _expected_new_pages(prefix_lens, seq_lens, 4)  # 5
        free_before = len(a.free_pages)

        out = a.alloc_extend(
            prefix_lens,
            prefix_lens.clone(),
            seq_lens,
            seq_lens.clone(),
            last_loc,
            extend_num_tokens=extend,
        )
        self.assertEqual(out.shape, (extend,))
        self.assertEqual(out.dtype, torch.int32)
        self.assertEqual(len(torch.unique(out)), extend)  # no slot reused
        self.assertGreaterEqual(int(out.min()), 0)
        self.assertLess(int(out.max()), 32)
        self.assertEqual(len(a.free_pages), free_before - pages)

    def test_extend_from_nonzero_prefix_continues_partial_page(self):
        a = _make_allocator(size=32, page_size=4)
        # prefix=2 (mid reserved page 0), extend to seq=6. last_loc=1 so the
        # partial page is filled first (slots 2,3), then one fresh page (4,5).
        prefix_lens = torch.tensor([2], dtype=torch.int32)
        seq_lens = torch.tensor([6], dtype=torch.int32)
        last_loc = torch.tensor([1], dtype=torch.int32)

        out = a.alloc_extend(
            prefix_lens,
            prefix_lens.clone(),
            seq_lens,
            seq_lens.clone(),
            last_loc,
            extend_num_tokens=4,
        )
        self.assertTrue(torch.equal(out, torch.tensor([2, 3, 4, 5], dtype=torch.int32)))
        self.assertEqual(len(a.free_pages), 8 - 1)

    def test_extend_page_aligned_prefix_exact_indices(self):
        a = _make_allocator(size=32, page_size=4)
        # prefix=0 -> no partial page; seq=8 -> exactly two fresh pages.
        prefix_lens = torch.tensor([0], dtype=torch.int32)
        seq_lens = torch.tensor([8], dtype=torch.int32)
        last_loc = torch.tensor([-1], dtype=torch.int32)

        out = a.alloc_extend(
            prefix_lens,
            prefix_lens.clone(),
            seq_lens,
            seq_lens.clone(),
            last_loc,
            extend_num_tokens=8,
        )
        self.assertTrue(
            torch.equal(
                out, torch.tensor([4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.int32)
            )
        )
        self.assertEqual(len(a.free_pages), 8 - 2)

    def test_extend_nondefault_page_size_eight(self):
        a = _make_allocator(size=64, page_size=8)
        prefix_lens = torch.tensor([0], dtype=torch.int32)
        seq_lens = torch.tensor([8], dtype=torch.int32)
        last_loc = torch.tensor([-1], dtype=torch.int32)

        out = a.alloc_extend(
            prefix_lens,
            prefix_lens.clone(),
            seq_lens,
            seq_lens.clone(),
            last_loc,
            extend_num_tokens=8,
        )
        self.assertTrue(
            torch.equal(
                out,
                torch.tensor([8, 9, 10, 11, 12, 13, 14, 15], dtype=torch.int32),
            )
        )
        self.assertEqual(len(a.free_pages), 8 - 1)

    def test_extend_varied_inputs_preserve_invariants(self):
        # Sweep page sizes / lengths / batch sizes (all prefix=0 so each
        # request allocates only fresh pages — no cross-request overlap in
        # the shared reserved page). Assert invariants that hold for ANY
        # valid extend allocation.
        configs = [
            (4, [3]),
            (4, [9]),
            (4, [4, 5]),
            (4, [4, 8, 5]),
            (4, [4, 9, 6, 11, 7]),
            (8, [8, 20]),
            (2, [3, 5, 6]),
        ]
        for page_size, seq in configs:
            with self.subTest(page_size=page_size, seq=seq):
                a = _make_allocator(size=256, page_size=page_size)
                prefix_t = torch.zeros(len(seq), dtype=torch.int32)
                seq_t = torch.tensor(seq, dtype=torch.int32)
                last_loc = torch.full((len(seq),), -1, dtype=torch.int32)
                extend = int(seq_t.sum())
                pages = _expected_new_pages(prefix_t, seq_t, page_size)
                free_before = len(a.free_pages)

                out = a.alloc_extend(
                    prefix_t,
                    prefix_t.clone(),
                    seq_t,
                    seq_t.clone(),
                    last_loc,
                    extend_num_tokens=extend,
                )
                self.assertIsNotNone(out)
                self.assertEqual(out.shape, (extend,))
                self.assertEqual(out.dtype, torch.int32)
                self.assertEqual(len(torch.unique(out)), extend)  # unique slots
                self.assertGreaterEqual(int(out.min()), 0)
                self.assertLess(int(out.max()), 256)
                self.assertEqual(len(a.free_pages), free_before - pages)


class TestNeedSortMergePath(CustomTestCase):
    """``need_sort=True`` (PD-disaggregation mode) routes frees into
    ``release_pages`` and defers consolidation to ``merge_and_sort_free``.
    The default ``need_sort=False`` suites never touch this path; these
    exercise it on CPU (sort correctness + the auto-merge-when-short guard
    in ``alloc`` / ``alloc_extend``)."""

    def test_need_sort_routes_free_to_release_pages(self):
        a = _make_allocator(size=32, page_size=4, need_sort=True)
        out = a.alloc(8)  # consumes pages 1, 2
        self.assertEqual(len(a.free_pages), 6)
        self.assertEqual(len(a.release_pages), 0)

        a.free(out)
        # Frees are staged in release_pages, NOT merged into free_pages yet.
        self.assertEqual(len(a.free_pages), 6)
        self.assertEqual(len(a.release_pages), 2)

    def test_merge_and_sort_restores_sorted_pool(self):
        a = _make_allocator(size=32, page_size=4, need_sort=True)
        out = a.alloc(8)  # pages 1,2 -> free_pages = [3,4,5,6,7,8]
        a.free(out)  # release_pages = [1,2]
        # A plain cat would give [3,4,5,6,7,8,1,2] (unsorted); merge must sort.
        a.merge_and_sort_free()

        self.assertEqual(len(a.release_pages), 0)
        self.assertTrue(
            torch.equal(a.free_pages, torch.arange(1, 9, dtype=torch.int32))
        )
        # Pools stay host-resident int32 (parent recreates release_pages on
        # free_pages.device, which is CPU for this subclass).
        self.assertEqual(a.free_pages.device.type, "cpu")
        self.assertEqual(a.release_pages.device.type, "cpu")
        self.assertEqual(a.free_pages.dtype, torch.int32)
        self.assertEqual(a.release_pages.dtype, torch.int32)

    def test_alloc_auto_merges_when_free_pages_short(self):
        a = _make_allocator(size=32, page_size=4, need_sort=True)
        first = a.alloc(16)  # pages 1..4 -> free_pages = [5,6,7,8]
        a.free(first)  # release_pages = [1,2,3,4]; free_pages = [5,6,7,8]
        self.assertEqual(len(a.free_pages), 4)
        self.assertEqual(len(a.release_pages), 4)

        # Needs 6 pages but only 4 in free_pages -> alloc must merge first.
        out = a.alloc(24)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, (24,))
        self.assertEqual(len(a.release_pages), 0)  # merged
        self.assertEqual(len(a.free_pages), 2)  # 8 - 6 consumed

    def test_alloc_extend_auto_merges_when_free_pages_short(self):
        a = _make_allocator(size=32, page_size=4, need_sort=True)
        first = a.alloc(16)  # pages 1..4 -> free_pages = [5,6,7,8]
        a.free(first)  # release_pages = [1,2,3,4]
        # Extend needing 6 fresh pages while free_pages holds 4 -> merge, succeed.
        prefix = torch.tensor([0], dtype=torch.int32)
        seq = torch.tensor([24], dtype=torch.int32)
        out = a.alloc_extend(
            prefix,
            prefix.clone(),
            seq,
            seq.clone(),
            torch.tensor([-1], dtype=torch.int32),
            extend_num_tokens=24,
        )
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, (24,))
        self.assertEqual(len(a.release_pages), 0)
        self.assertEqual(len(a.free_pages), 2)

    def test_decode_auto_merges_when_free_pages_short(self):
        # alloc_decode calls merge_and_sort_free unconditionally when short.
        a = _make_allocator(size=32, page_size=4, need_sort=True)
        first = a.alloc(28)  # pages 1..7 -> free_pages = [8]
        a.free(first)  # release_pages = [1..7]; free_pages = [8]
        self.assertEqual(len(a.free_pages), 1)
        self.assertEqual(len(a.release_pages), 7)

        # Two reqs both crossing a boundary need 2 pages; free_pages has 1.
        out = a.alloc_decode(
            torch.tensor([5, 9], dtype=torch.int32),
            torch.tensor([5, 9], dtype=torch.int32),
            torch.tensor([15, 23], dtype=torch.int32),
        )
        self.assertIsNotNone(out)
        self.assertEqual(out.shape, (2,))
        self.assertEqual(len(a.release_pages), 0)  # merged
        self.assertEqual(len(a.free_pages), 8 - 2)


@unittest.skipUnless(_TPU_AVAILABLE, "requires torch_tpu and a usable TPU device")
class TestAllocatorOnTpuDevice(CustomTestCase):
    """End-to-end checks on a real ``tpu`` device — the half the CPU tests
    cannot reach: the H2D copy + device formula round-trip, ``non_blocking``
    transfer lifetime, output device/dtype, and (critically) that the
    host-resident pools never migrate onto the accelerator.

    Strategy: run the same logical inputs through a CPU allocator and a TPU
    allocator built identically, then assert the device output (moved back
    with ``.cpu()``) equals the CPU reference. A correct H2D->formula->D2H
    round-trip is the only way they can match.
    """

    def test_decode_device_matches_cpu(self):
        # Mixed batch: continue / new / continue / new. last_loc values
        # satisfy the decode invariant ((last_loc+2) % page_size == seq %
        # page_size); the new-page entries are page-boundary slots.
        seq = [3, 5, 7, 9]
        last_loc = [9, 15, 13, 19]
        cpu_a = _make_allocator(size=64, page_size=4, device="cpu")
        tpu_a = _make_allocator(size=64, page_size=4, device="tpu")

        out_cpu = cpu_a.alloc_decode(
            torch.tensor(seq, dtype=torch.int32),
            torch.tensor(seq, dtype=torch.int32),
            torch.tensor(last_loc, dtype=torch.int32),
        )
        out_tpu = tpu_a.alloc_decode(
            torch.tensor(seq, dtype=torch.int32, device="tpu"),
            torch.tensor(seq, dtype=torch.int32),
            torch.tensor(last_loc, dtype=torch.int32, device="tpu"),
        )
        self.assertEqual(out_tpu.device.type, "tpu")
        self.assertEqual(out_tpu.dtype, torch.int32)
        self.assertTrue(torch.equal(out_tpu.cpu(), out_cpu))

    def test_extend_device_matches_cpu(self):
        prefix = [0, 0, 0]
        seq = [4, 8, 5]
        extend = sum(seq)
        cpu_a = _make_allocator(size=64, page_size=4, device="cpu")
        tpu_a = _make_allocator(size=64, page_size=4, device="tpu")

        out_cpu = cpu_a.alloc_extend(
            torch.tensor(prefix, dtype=torch.int32),
            torch.tensor(prefix, dtype=torch.int32),
            torch.tensor(seq, dtype=torch.int32),
            torch.tensor(seq, dtype=torch.int32),
            torch.tensor([-1, -1, -1], dtype=torch.int32),
            extend_num_tokens=extend,
        )
        out_tpu = tpu_a.alloc_extend(
            torch.tensor(prefix, dtype=torch.int32, device="tpu"),
            torch.tensor(prefix, dtype=torch.int32),
            torch.tensor(seq, dtype=torch.int32, device="tpu"),
            torch.tensor(seq, dtype=torch.int32),
            torch.tensor([-1, -1, -1], dtype=torch.int32, device="tpu"),
            extend_num_tokens=extend,
        )
        self.assertEqual(out_tpu.device.type, "tpu")
        self.assertEqual(out_tpu.dtype, torch.int32)
        self.assertTrue(torch.equal(out_tpu.cpu(), out_cpu))

    def test_alloc_device_matches_cpu(self):
        cpu_a = _make_allocator(size=32, page_size=4, device="cpu")
        tpu_a = _make_allocator(size=32, page_size=4, device="tpu")
        out_cpu = cpu_a.alloc(8)
        out_tpu = tpu_a.alloc(8)
        self.assertEqual(out_tpu.device.type, "tpu")
        self.assertEqual(out_tpu.dtype, torch.int32)
        self.assertTrue(torch.equal(out_tpu.cpu(), out_cpu))

    def test_pools_stay_on_host(self):
        # Even with device="tpu", the free/release pools are host bookkeeping
        # and must never end up on the accelerator.
        a = _make_allocator(size=32, page_size=4, device="tpu")
        a.alloc_decode(
            torch.tensor([5, 9], dtype=torch.int32, device="tpu"),
            torch.tensor([5, 9], dtype=torch.int32),
            torch.tensor([15, 23], dtype=torch.int32, device="tpu"),
        )
        self.assertEqual(a.free_pages.device.type, "cpu")
        self.assertEqual(a.release_pages.device.type, "cpu")

    def test_merge_keeps_release_pages_on_host(self):
        # need_sort=True routes frees into release_pages. The parent's
        # merge_and_sort_free defaults to self.device (the accelerator), so
        # the subclass passes device="cpu" to keep the host-resident pools on
        # host — verify release_pages stays on CPU and free()-after-merge
        # works (a cpu release_pages cat'd with cpu page indices).
        a = _make_allocator(size=32, page_size=4, device="tpu", need_sort=True)
        out = a.alloc(8)
        a.free(out)
        self.assertEqual(a.release_pages.device.type, "cpu")

        # Mirror the subclass's own call convention.
        a.merge_and_sort_free(device="cpu")
        self.assertEqual(a.free_pages.device.type, "cpu")
        self.assertEqual(a.release_pages.device.type, "cpu")
        # Pool fully restored and sorted.
        self.assertEqual(len(a.free_pages), 8)
        self.assertTrue(torch.equal(a.free_pages, torch.sort(a.free_pages).values))

        # free() after a merge must not hit a cpu/tpu device mismatch.
        out2 = a.alloc(8)
        a.free(out2)
        self.assertEqual(a.release_pages.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
