"""CPU ownership checks across TPU allocation, exhaustion, and grouped release."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator import swa
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.allocator_tpu import (
    TpuPagedTokenToKVPoolAllocator,
)
from sglang_tpu.kv_free_fence import install_kv_free_fence
from sglang_tpu.tpu_platform import TpuSRTPlatform

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _TpuIndices(torch.Tensor):
    """Host values that report a non-CPU device, as TPU indices would."""

    @property
    def device(self):
        return torch.device("meta")


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

    def test_swa_segment_frees_need_host_indices(self):
        """Segment frees index the host-resident full-to-SWA mapping directly:
        host indices free cleanly, and TPU indices fail with a clear message
        instead of an indexing error inside the free."""
        page_size, size = 4, 16
        frees = {
            "free_segment": lambda a, idx: a.free_segment(idx, start_pos=0),
            "free_full_segment": lambda a, idx: (
                a.free_swa_segment(idx, start_pos=0),
                a.free_full_segment(idx, start_pos=0),
            ),
        }
        for name, free in frees.items():
            with (
                self.subTest(free=name),
                patch.object(swa, "_is_tpu", True),
                patch.object(swa, "current_platform", TpuSRTPlatform()),
                patch.object(
                    swa,
                    "TpuPagedTokenToKVPoolAllocator",
                    TpuPagedTokenToKVPoolAllocator,
                    create=True,
                ),
            ):
                kv_pool = SWAKVPool(
                    size=size,
                    size_swa=size,
                    page_size=page_size,
                    dtype=torch.bfloat16,
                    head_num=1,
                    head_dim=8,
                    swa_attention_layer_ids=[1],
                    full_attention_layer_ids=[0],
                    device="cpu",
                )
                allocator = swa.SWATokenToKVPoolAllocator(
                    size=size,
                    size_swa=size,
                    page_size=page_size,
                    dtype=torch.bfloat16,
                    device="cpu",
                    kvcache=kv_pool,
                    need_sort=False,
                )
                zero = torch.zeros(1, dtype=torch.int64)
                seq = torch.tensor([2 * page_size], dtype=torch.int64)
                slots = allocator.alloc_extend(
                    zero, zero, seq, seq, zero - 1, 2 * page_size
                ).long()

                with self.assertRaisesRegex(
                    AssertionError, "SGLANG_TPU_REQ_TO_TOKEN_ON_HOST"
                ):
                    free(allocator, slots.as_subclass(_TpuIndices))
                free(allocator, slots)

                self.assertEqual(allocator.full_available_size(), size)
                self.assertEqual(allocator.swa_available_size(), size)
                self.assertFalse(allocator.full_to_swa_index_mapping[:-1].any())

    def _allocator(self, *, need_sort, page_size=4, num_pages=8):
        return TpuPagedTokenToKVPoolAllocator(
            size=num_pages * page_size,
            page_size=page_size,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=need_sort,
        )

    def test_releasing_a_free_page_again_raises(self):
        """A page freed in two separate calls would sit twice in the pools and
        the next allocation would hand it to two owners."""
        page_size = 4
        for need_sort in (False, True):
            frees = {
                "free": lambda a, idx: a.free(idx),
                "free_segment": lambda a, idx: a.free_segment(idx, start_pos=0),
                "free_page_ids": lambda a, idx: a.free_page_ids(
                    idx[::page_size] // page_size
                ),
            }
            for first_name, first in frees.items():
                for second_name, second in frees.items():
                    with self.subTest(
                        need_sort=need_sort, first=first_name, second=second_name
                    ):
                        allocator = self._allocator(need_sort=need_sort)
                        slots = allocator.alloc(2 * page_size).long()
                        other = allocator.alloc(page_size).long()
                        first(allocator, slots)
                        page = int(slots[0]) // page_size
                        with self.assertRaisesRegex(
                            RuntimeError, f"already free: \\[{page}, "
                        ):
                            second(allocator, slots)
                        # The second call changed nothing: one copy of each page.
                        self._check_ownership(allocator, [other])
                        reused = allocator.alloc(7 * page_size)
                        self.assertEqual(reused.numel(), torch.unique(reused).numel())

    def test_releasing_a_held_page_again_raises(self):
        """Under the KV free fence a freed page waits in held_pages while a
        forward is in flight; a second release must not slip past that."""
        page_size = 4
        allocator = self._allocator(need_sort=True)
        fence = install_kv_free_fence(
            allocator,
            synchronize_device=lambda: None,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            copies_draft_tail=False,
        )
        slots = allocator.alloc(page_size).long()
        fence.on_batch_formed()
        fence.on_forward_launched(SimpleNamespace(copy_done=None))
        allocator.free(slots)
        self.assertEqual(allocator.held_pages.num_pages, 1)
        with self.assertRaisesRegex(RuntimeError, "already free"):
            allocator.free(slots)
        self.assertEqual(allocator.held_pages.num_pages, 1)

    def test_a_page_listed_twice_in_one_release_raises(self):
        """free_page_ids and free_segment take each page once; a repeat would
        enter the pools twice like a cross-call double release."""
        page_size = 4
        allocator = self._allocator(need_sort=False)
        slots = allocator.alloc(2 * page_size).long()
        pages = slots[::page_size] // page_size
        with self.assertRaisesRegex(RuntimeError, "twice in one call"):
            allocator.free_page_ids(torch.cat((pages, pages[:1])))
        allocator.free_page_ids(pages)
        self._check_ownership(allocator, [])

    def test_handing_out_a_page_twice_raises(self):
        """A duplicate that reached the free list by any other route is caught
        when it is handed out, not after two requests share the page."""
        page_size = 4
        allocator = self._allocator(need_sort=False)
        slots = allocator.alloc(page_size)
        allocator.free_pages = torch.cat(
            (allocator.free_pages, (slots[:1] // page_size).to(torch.int32))
        )
        with self.assertRaisesRegex(RuntimeError, "not free"):
            allocator.alloc(allocator.available_size())

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
