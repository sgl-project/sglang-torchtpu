"""CPU ownership checks across TPU allocation, exhaustion, and grouped release."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.allocator_tpu import (
    TpuPagedTokenToKVPoolAllocator,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTpuAllocatorLifecycle(CustomTestCase):
    def test_direct_page_release_preserves_decode_dtype(self):
        page_size = 4
        allocator = TpuPagedTokenToKVPoolAllocator(
            size=6 * page_size,
            page_size=page_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        allocated = allocator.alloc(page_size)

        allocator.free_page_ids((allocated[::page_size] // page_size).long())

        self.assertEqual(allocator.free_pages.dtype, torch.int32)
        out = allocator.alloc_decode(
            torch.tensor([5], dtype=torch.int32),
            torch.tensor([5], dtype=torch.int32),
            torch.tensor([15], dtype=torch.int32),
        )
        self.assertIsNotNone(out)
        self.assertEqual(out.dtype, torch.int32)

    def test_exhaustion_grouped_release_and_retry_preserve_ownership(self):
        for page_size in (4, 8, 32):
            for need_sort in (False, True):
                with self.subTest(page_size=page_size, need_sort=need_sort):
                    allocator = TpuPagedTokenToKVPoolAllocator(
                        size=6 * page_size,
                        page_size=page_size,
                        dtype=torch.bfloat16,
                        device="cpu",
                        kvcache=None,
                        need_sort=need_sort,
                    )
                    # Both requests own a partial page, then grow across it.
                    lengths = torch.tensor([page_size - 1, 1], dtype=torch.int64)
                    zero = torch.zeros(2, dtype=torch.int64)
                    first = allocator.alloc_extend(
                        zero, zero, lengths, lengths, zero - 1, int(lengths.sum())
                    )
                    requests = list(first.split(lengths.tolist()))
                    self._check_ownership(allocator, requests)

                    grown = lengths + page_size
                    added = allocator.alloc_extend(
                        lengths,
                        lengths,
                        grown,
                        grown,
                        torch.stack([row[-1] for row in requests]),
                        2 * page_size,
                    )
                    for old, new in zip(requests, added.split(page_size)):
                        self.assertEqual(int(new[0]), int(old[-1]) + 1)
                    requests = [
                        torch.cat((old, new))
                        for old, new in zip(requests, added.split(page_size))
                    ]
                    self._check_ownership(allocator, requests)

                    # An unsatisfied request must leave every existing owner intact.
                    self.assertIsNone(allocator.alloc(3 * page_size))
                    self._check_ownership(allocator, requests)
                    saved = [row.clone() for row in requests]
                    allocator.free_group_begin()
                    allocator.free_segment(requests[0].long(), start_pos=0)
                    allocator.free(requests[1].long())
                    self.assertEqual(allocator.available_size(), 2 * page_size)
                    allocator.free_group_end()
                    self._check_ownership(allocator, [])
                    for before, after in zip(saved, requests):
                        self.assertTrue(torch.equal(before, after))

                    # Reuse all released pages, then cross a decode page boundary.
                    full = allocator.alloc(6 * page_size)
                    self._check_ownership(allocator, [full])
                    seq = torch.tensor([page_size + 1], dtype=torch.int64)
                    last = full[page_size - 1 : page_size]
                    self.assertIsNone(allocator.alloc_decode(seq, seq, last))
                    self._check_ownership(allocator, [full])
                    allocator.free_segment(full[page_size:].long(), start_pos=page_size)
                    live = full[:page_size]
                    token = allocator.alloc_decode(seq, seq, last)
                    self.assertEqual(token.numel(), 1)
                    self.assertEqual(int(token[0]) % page_size, 0)
                    self._check_ownership(allocator, [torch.cat((live, token))])

    def _check_ownership(self, allocator, requests):
        owners = [set((row // allocator.page_size).tolist()) for row in requests]
        occupied = set().union(*owners)
        self.assertEqual(sum(map(len, owners)), len(occupied))
        for row in requests:
            self.assertEqual(row.numel(), torch.unique(row).numel())
        free = allocator.get_all_free_pages()
        self.assertEqual(free.dtype, torch.int32)
        self.assertEqual(free.device.type, "cpu")
        available = set(free.tolist())
        self.assertEqual(free.numel(), len(available))
        self.assertFalse(occupied & available)
        self.assertEqual(occupied | available, set(range(1, allocator.num_pages + 1)))
        self.assertEqual(
            allocator.available_size(), len(available) * allocator.page_size
        )


if __name__ == "__main__":
    unittest.main()
