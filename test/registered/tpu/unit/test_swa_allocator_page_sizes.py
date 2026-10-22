"""``SWATokenToKVPoolAllocator`` with a sliding-window side paged finer than
the full side (``swa_page_size``): an unsplit SWA sub-pool under decode
context parallelism keeps the physical page while the full side's locations
are widened to ``page_size * dcp``. Host-only logic over the TPU paged
sub-allocators, so every case runs on the CPU."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator import swa as swa_module
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator

PS = 8  # the physical page (SWA side)
FULL_PS = 2 * PS  # the widened full page
FULL_ROWS = 16 * FULL_PS
SWA_ROWS = 12 * PS


def _t(values):
    return torch.tensor(values, dtype=torch.int64)


class _FakeSwaPool(BaseSWAKVPool):
    def __init__(self):
        self.full_kv_pool = None
        self.swa_kv_pool = None
        self.swa_req_ring_size = None
        self.full_to_swa_index_mapping = None

    def register_mapping(self, mapping):
        self.full_to_swa_index_mapping = mapping

    def translate_loc_from_full_to_swa(self, kv_indices):
        return self.full_to_swa_index_mapping[kv_indices.to(torch.int64)]

    def _unused(self, *args, **kwargs):
        raise NotImplementedError("the allocator never reads KV buffers")

    get_key_buffer = get_value_buffer = get_kv_buffer = _unused
    get_state_buf_infos = set_kv_buffer = _unused


class TestSwaPageSize(CustomTestCase):
    def setUp(self):
        # The composite picks the TPU sub-allocators at import and call time.
        for patcher in (
            patch.object(swa_module, "_is_tpu", True),
            patch.object(
                swa_module,
                "TpuPagedTokenToKVPoolAllocator",
                TpuPagedTokenToKVPoolAllocator,
                create=True,
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _make(self, swa_page_size=PS):
        return swa_module.SWATokenToKVPoolAllocator(
            FULL_ROWS,
            SWA_ROWS,
            page_size=FULL_PS,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=_FakeSwaPool(),
            need_sort=False,
            swa_page_size=swa_page_size,
        )

    def _extend(self, alloc, prefix, seq, last_loc):
        return alloc.alloc_extend(
            prefix_lens=_t([prefix]),
            prefix_lens_cpu=_t([prefix]),
            seq_lens=_t([seq]),
            seq_lens_cpu=_t([seq]),
            last_loc=_t([last_loc]),
            extend_num_tokens=seq - prefix,
        )

    def _check_mapping(self, alloc, locs):
        """Every full location maps to an SWA slot whose in-page offset is the
        position's offset at the SWA page, and SWA slots are distinct."""
        mapping = alloc.full_to_swa_index_mapping
        swa = mapping[locs.to(torch.int64)]
        self.assertTrue(torch.all(swa > 0))
        positions = torch.arange(locs.numel())
        self.assertTrue(torch.equal(swa % PS, positions % PS))
        self.assertEqual(len(torch.unique(swa)), locs.numel())

    def test_geometry(self):
        alloc = self._make()
        self.assertEqual((alloc.page_size, alloc.swa_page_size), (FULL_PS, PS))
        self.assertEqual(alloc.full_attn_allocator.page_size, FULL_PS)
        self.assertEqual(alloc.swa_attn_allocator.page_size, PS)
        self.assertEqual(alloc.swa_attn_allocator.num_pages, SWA_ROWS // PS)
        self.assertEqual(
            alloc.full_to_swa_index_mapping.numel(), FULL_ROWS + FULL_PS + 1
        )
        self.assertEqual(alloc.swa_available_size(), SWA_ROWS)
        self.assertEqual(alloc.full_available_size(), FULL_ROWS)

    def test_extend_pages_each_side_at_its_own_page(self):
        alloc = self._make()
        locs = self._extend(alloc, 0, 3 * PS, -1).to(torch.int64)
        # 3 SWA pages (24 tokens) but 2 widened full pages.
        self.assertEqual(alloc.swa_available_size(), SWA_ROWS - 3 * PS)
        self.assertEqual(alloc.full_available_size(), FULL_ROWS - 2 * FULL_PS)
        self._check_mapping(alloc, locs)
        # Continue inside the second full page but into a fourth SWA page.
        more = self._extend(alloc, 3 * PS, 3 * PS + 3, int(locs[-1])).to(torch.int64)
        locs = torch.cat([locs, more])
        self.assertEqual(alloc.swa_available_size(), SWA_ROWS - 4 * PS)
        self.assertEqual(alloc.full_available_size(), FULL_ROWS - 2 * FULL_PS)
        self._check_mapping(alloc, locs)
        alloc.free(locs)
        self.assertEqual(alloc.swa_available_size(), SWA_ROWS)
        self.assertEqual(alloc.full_available_size(), FULL_ROWS)
        self.assertTrue(torch.all(alloc.full_to_swa_index_mapping[:-1] == 0))

    def test_decode_opens_an_swa_page_inside_a_full_page(self):
        alloc = self._make()
        locs = self._extend(alloc, 0, PS, -1).to(torch.int64)
        out = alloc.alloc_decode(_t([PS + 1]), _t([PS + 1]), locs[-1:]).to(torch.int64)
        locs = torch.cat([locs, out])
        # The full side continues its page; the SWA side opened page 2.
        self.assertEqual(int(out[0]), int(locs[-2]) + 1)
        self.assertEqual(alloc.full_available_size(), FULL_ROWS - FULL_PS)
        self.assertEqual(alloc.swa_available_size(), SWA_ROWS - 2 * PS)
        self._check_mapping(alloc, locs)

    def test_prealloc_tail_counts_swa_pages_at_the_swa_page(self):
        alloc = self._make()
        fill_len, tail = 3 * PS + 2, PS + 2  # tail starts at 2 PS, a full page start
        locs = alloc.alloc_extend_swa_tail(
            prefix_lens=_t([0]),
            prefix_lens_cpu=_t([0]),
            seq_lens=_t([fill_len]),
            seq_lens_cpu=_t([fill_len]),
            last_loc=_t([-1]),
            extend_num_tokens=fill_len,
            swa_tail_len=tail,
        ).to(torch.int64)
        self.assertEqual(alloc.swa_available_size(), SWA_ROWS - 2 * PS)
        mapping = alloc.full_to_swa_index_mapping
        self.assertTrue(torch.all(mapping[locs[:-tail]] == 0))
        self._check_mapping(alloc, locs[-tail:])
        alloc.free(locs)
        self.assertEqual(alloc.swa_available_size(), SWA_ROWS)

    def test_segment_free_releases_every_swa_page(self):
        alloc = self._make()
        locs = self._extend(alloc, 0, 2 * FULL_PS, -1).to(torch.int64)
        self.assertEqual(alloc.swa_available_size(), SWA_ROWS - 4 * PS)
        alloc.free_segment(locs, start_pos=0)
        self.assertEqual(alloc.swa_available_size(), SWA_ROWS)
        self.assertEqual(alloc.full_available_size(), FULL_ROWS)

    def test_budget_and_resize_use_the_swa_page(self):
        alloc = self._make()
        self.assertTrue(alloc.new_pages_available(1, SWA_ROWS // PS))
        self.assertFalse(alloc.new_pages_available(1, SWA_ROWS // PS + 1))
        alloc.resize(
            SimpleNamespace(
                full_max_total_num_tokens=4 * FULL_PS, swa_max_total_num_tokens=5 * PS
            )
        )
        self.assertEqual(alloc.swa_attn_allocator.num_pages, 5)
        self.assertEqual(alloc.full_attn_allocator.num_pages, 4)

    def _req(self, committed, allocated=None):
        allocated = committed if allocated is None else allocated
        return SimpleNamespace(
            kv=SimpleNamespace(kv_committed_len=committed, kv_allocated_len=allocated)
        )

    def _running_and_room(self, alloc, room_len):
        """The decode node's mixed state: one running request whose allocation
        is aligned to the full page, and one preallocated room (a transferred
        request on its first decode steps, allocated through the SWA-tail
        prealloc) whose length is not; the room's tail takes every SWA page
        that is left while the full side keeps free pages."""
        running = self._extend(alloc, 0, 2 * FULL_PS, -1).to(torch.int64)
        tail = alloc.swa_available_size()
        self.assertLessEqual(tail, room_len)
        room = alloc.alloc_extend_swa_tail(
            prefix_lens=_t([0]),
            prefix_lens_cpu=_t([0]),
            seq_lens=_t([room_len]),
            seq_lens_cpu=_t([room_len]),
            last_loc=_t([-1]),
            extend_num_tokens=room_len,
            swa_tail_len=tail,
        ).to(torch.int64)
        self.assertEqual(alloc.swa_available_size(), 0)
        self.assertGreater(alloc.full_available_size(), 2 * FULL_PS)
        return running, room

    def test_decode_capacity_sees_the_swa_page_a_plain_decode_opens(self):
        """Room at 72 = 4.5 full pages, on an SWA page boundary: the batch
        counts no full page for the next token, which opens an SWA page."""
        alloc = self._make()
        running, room = self._running_and_room(alloc, room_len=9 * PS)
        reqs = [self._req(2 * FULL_PS - 1, allocated=2 * FULL_PS), self._req(9 * PS)]
        self.assertFalse(
            alloc.check_decode_capacity(
                num_tokens=0, tree_cache=None, requests=reqs, spec_algorithm=None
            )
        )
        # The step's allocation agrees with the gate.
        self.assertIsNone(
            alloc.alloc_decode(_t([9 * PS + 1]), _t([9 * PS + 1]), room[-1:])
        )
        # One SWA page back (the running request moves on) and the step fits.
        alloc.free_segment(running[:PS], start_pos=0)
        self.assertTrue(
            alloc.check_decode_capacity(
                num_tokens=0, tree_cache=None, requests=reqs, spec_algorithm=None
            )
        )

    def test_decode_capacity_sees_the_swa_page_a_spec_step_opens(self):
        """Room at 67: the speculative reserve rounds the allocation up to 80,
        inside full page 64..79 (no full page for the batch to count), while
        SWA page 72..79 is still unopened."""
        alloc = self._make()
        room_len = 4 * FULL_PS + 3
        running, room = self._running_and_room(alloc, room_len=room_len)
        reqs = [self._req(2 * FULL_PS - 2, allocated=2 * FULL_PS), self._req(room_len)]
        spec = SimpleNamespace(
            is_some=lambda: True, is_none=lambda: False, is_dflash_family=lambda: False
        )
        with (
            patch.object(swa_module, "get_alloc_reserve_per_decode", lambda: 2),
            patch.object(
                swa_module,
                "get_schedule",
                lambda: SimpleNamespace(disable_overlap_schedule=True),
            ),
        ):
            # Full: ceil16(80) - ceil16(67) = 0 pages; SWA: ceil8(80) - ceil8(67) = 1.
            self.assertFalse(
                alloc.check_decode_capacity(
                    num_tokens=0, tree_cache=None, requests=reqs, spec_algorithm=spec
                )
            )
            self.assertIsNone(self._extend(alloc, room_len, 5 * FULL_PS, int(room[-1])))
            # A room already on the full-page grid needs both sides equally.
            self.assertTrue(
                alloc.check_decode_capacity(
                    num_tokens=0,
                    tree_cache=None,
                    requests=[
                        reqs[0],
                        self._req(5 * FULL_PS - 4, allocated=5 * FULL_PS),
                    ],
                    spec_algorithm=spec,
                )
            )
        # Without request context the gate keeps today's single count.
        self.assertTrue(alloc.check_decode_capacity(num_tokens=0, tree_cache=None))

    def test_default_single_page_unchanged(self):
        alloc = self._make(swa_page_size=None)
        self.assertEqual(alloc.swa_page_size, FULL_PS)
        self.assertEqual(alloc.swa_attn_allocator.page_size, FULL_PS)
        locs = self._extend(alloc, 0, 3 * PS, -1).to(torch.int64)
        self.assertEqual(alloc.swa_available_size(), SWA_ROWS - 2 * FULL_PS)
        self._check_mapping(alloc, locs)


if __name__ == "__main__":
    unittest.main()
