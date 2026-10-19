"""A PD prefill send never lists a page the radix tree can still reclaim.

A checkpoint capped at ``swa_branching_seqlen`` (hybrid SWA) publishes less
than the chunk; the unpublished positions must not be sent until the next
checkpoint owns them, or its dedup frees pages a transfer still lists and the
closing registration repeats the page raiden then rejects.

The scenario runs the real FULL+SWA unified tree over real TPU page allocators
and the real ``_send_kv_chunk`` in the overlap loop's order: chunk k's
checkpoint runs when batch k+1 forms, before chunk k's send.
"""

import unittest
from array import array
from types import SimpleNamespace
from typing import List
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator import swa
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.mem_cache.allocator.allocator_tpu import TpuPagedTokenToKVPoolAllocator
from sglang_tpu.platforms.tpu_platform import TpuSRTPlatform

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 4
WINDOW = 8


def ids(seed, n):
    return [1000 * seed + i for i in range(n)]


POOL_TOKENS = 32 * PAGE
TOKENS = list(range(1, 49))  # 12 pages
ROOM = 7


class _Sender:
    """Records the room's send list the way RaidenKVManager accumulates it."""

    def __init__(self, blocks: List[int]):
        self.blocks = blocks

    def get_max_transfer_tokens(self):
        return None

    def should_send_kv_chunk(self, num_pages: int, last_chunk: bool) -> bool:
        return num_pages > 0

    def send(self, page_indices, state_indices=None, num_kv_tokens=None):
        self.blocks.extend(int(p) for p in page_indices)


class _Scheduler(SchedulerDisaggregationPrefillMixin):
    def __init__(self, allocator, req_to_token_pool, tree_cache):
        self.token_to_kv_pool_allocator = allocator
        self.req_to_token_pool = req_to_token_pool
        self.tree_cache = tree_cache
        self.enable_overlap = True
        self.enable_staging = False
        self.kv_checksum_computer = None


class TestChunkSendsStayOwned(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", page_size=PAGE), role="scheduler")
        with (
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
                size=POOL_TOKENS,
                size_swa=POOL_TOKENS,
                page_size=PAGE,
                dtype=torch.bfloat16,
                head_num=1,
                head_dim=8,
                swa_attention_layer_ids=[1],
                full_attention_layer_ids=[0],
                device="cpu",
            )
            # A PD prefill engine sorts freed pages; the mechanism only needs
            # the freed page to come back to the same request, which the
            # prepend of the unsorted pool makes deterministic.
            self.allocator = swa.SWATokenToKVPoolAllocator(
                size=POOL_TOKENS,
                size_swa=POOL_TOKENS,
                page_size=PAGE,
                dtype=torch.bfloat16,
                device="cpu",
                kvcache=kv_pool,
                need_sort=False,
            )
        self.full = self.allocator.full_attn_allocator
        self.pool = ReqToTokenPool(
            size=4, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        self.cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=PAGE,
                sliding_window_size=WINDOW,
                eviction_policy="lru",
                tree_components=(ComponentType.FULL, ComponentType.SWA),
                tree_core_backend="python",
            )
        )
        self.scheduler = _Scheduler(self.allocator, self.pool, self.cache)
        self.blocks: List[int] = []

    # ---- request lifecycle ----------------------------------------------------

    def _req(self, rid, pool_idx, tokens):
        kv = ReqKvInfo(req_pool_idx=pool_idx)
        req = SimpleNamespace(
            rid=rid,
            origin_input_ids=list(tokens),
            output_ids=[],
            full_untruncated_fill_ids=array("q", tokens),
            extra_key=None,
            cache_salt=None,
            kv_rotation_base=None,
            session=None,
            session_id=None,
            priority=None,
            skip_radix_cache_insert=False,
            pending_bootstrap=False,
            host_hit_length=0,
            early_send_prefix_end=None,
            extend_range=None,
            start_send_idx=0,
            kv=kv,
            owned_kv_len=lambda: kv.kv_committed_len,
            finished=lambda: False,
            refresh_fill_ids=lambda: None,
            disagg_kv_sender=_Sender(self.blocks),
        )
        req.get_fill_ids = lambda: req.full_untruncated_fill_ids[: req.extend_range.end]
        return req

    def _match_and_lock(self, req):
        result = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", req.origin_input_ids)))
        )
        req.prefix_indices = result.device_indices
        req.last_node = result.last_device_node
        req.swa_branching_seqlen = result.swa_branching_seqlen
        req.kv.cache_protected_len = len(result.device_indices)
        req.lock_receipt = self.cache.inc_lock_ref(req.last_node).to_dec_params()
        req.swa_prefix_lock_released = False
        self.pool.req_to_token[req.kv.req_pool_idx, : len(req.prefix_indices)] = (
            req.prefix_indices
        )
        return result

    def _alloc_chunk(self, req, start, end):
        """prepare_for_extend for one request computing [start, end)."""
        prefix = torch.tensor([start], dtype=torch.int64)
        seq = torch.tensor([end], dtype=torch.int64)
        row = self.pool.req_to_token[req.kv.req_pool_idx]
        last_loc = torch.tensor([int(row[start - 1]) if start else -1])
        slots = self.allocator.alloc_extend(
            prefix, prefix, seq, seq, last_loc, end - start
        )
        self.assertIsNotNone(slots)
        row[start:end] = slots
        req.extend_range = SimpleNamespace(start=start, end=end)
        req.kv.kv_committed_len = req.kv.kv_allocated_len = end
        return slots.long()

    def _checkpoint(self, req, up_to):
        self.cache.checkpoint(req, up_to=up_to)

    def _send(self, req, end_idx):
        self.scheduler.send_kv_chunk(req, last_chunk=False, end_idx=end_idx)

    def _pages(self, slots):
        return (slots[::PAGE] // PAGE).tolist()

    def _free_pages(self):
        return set(self.full.get_all_free_pages().tolist())

    # ---- the scenario ---------------------------------------------------------

    def _seed_prefix_without_swa(self, tokens):
        """Cached Full KV for ``tokens`` whose SWA was evicted: two cached
        continuations make it an interior node, which keeps its Full KV when
        its SWA goes (a leaf would lose both)."""
        seed = self._req("seed", 3, tokens)
        for tail in (ids(7, 8), ids(8, 8)):
            slots = self._alloc_chunk(seed, 0, len(tokens) + 8)
            self.cache.insert(
                InsertParams(key=RadixKey(array("q", tokens + tail)), value=slots)
            )
        self.cache.evict(EvictParams(swa_num_tokens=len(tokens) + 16))

    def test_a_page_freed_under_a_pending_send_is_never_listed_twice(self):
        self._seed_prefix_without_swa(TOKENS[:16])

        # X has to recompute the prefix for its SWA, so it matches nothing; the
        # Full hit reaches 16, so the branch cap sits inside chunk 2.
        x = self._req("x", 1, TOKENS)
        match = self._match_and_lock(x)
        self.assertEqual(len(match.device_indices), 0)
        self.assertEqual(x.swa_branching_seqlen, 16)

        # Overlap loop, chunks of 12: form batch k+1 (checkpoint chunk k, then
        # allocate chunk k+1), then process batch k's result (send chunk k).
        self._alloc_chunk(x, 0, 12)
        self._checkpoint(x, 12)
        self.assertEqual(x.kv.cache_protected_len, 12)
        self._alloc_chunk(x, 12, 24)
        self._send(x, 12)

        self._checkpoint(x, 24)
        self.assertEqual(x.kv.cache_protected_len, 16)
        self._alloc_chunk(x, 24, 36)
        self._send(x, 24)
        own_tail = self.pool.req_to_token[1, 16:24].clone()

        # A sibling with the same tokens finishes and inserts [16, 24).
        y = self._req("y", 2, TOKENS[:24])
        self._match_and_lock(y)
        self.assertEqual(y.kv.cache_protected_len, 16)
        self._alloc_chunk(y, 16, 24)
        y.finished = lambda: True
        release_kv_cache(y, self.cache)

        # X's next checkpoint dedups [16, 24) against the sibling's leaf.
        self._checkpoint(x, 36)
        self.assertTrue(set(self._pages(own_tail)) <= self._free_pages())
        self.assertFalse(
            set(self.blocks) & self._free_pages(),
            "a page still listed for the room was freed",
        )
        self._alloc_chunk(x, 36, 48)
        self._send(x, 36)

        self._checkpoint(x, 48)
        self._send(x, 48)
        self.assertEqual(len(self.blocks), len(set(self.blocks)), self.blocks)
        # The send list covers the row the decode will be told about.
        self.assertEqual(self.blocks, self._pages(self.pool.req_to_token[1, :48]))


if __name__ == "__main__":
    unittest.main()
