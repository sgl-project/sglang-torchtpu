"""SWA tombstone recovery must start at the request's protected prefix.

A request owns FULL pages only from its protected prefix (``prev_prefix_len``)
upward; below it, its row holds the node's own canonical ids by reference. When
such a request re-inserts an SWA-tombstoned node and its SWA cursor sits below
the protected prefix (a single-chunk request has a cursor of 0), a recovery that
starts at the cursor treats the borrowed head as the request's duplicates and
frees pages the node still maps. The freed pages re-enter the free list while the
node keeps serving them, so the next allocation overwrites another request's KV.

Python mirror of the Rust core's clamp (sgl-project/sglang#46700).
"""

import unittest
from array import array

import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FULL = ComponentType.FULL
SWA = ComponentType.SWA
KV_SIZE = 64
WINDOW = 8


def _build_swa_cache() -> tuple[UnifiedRadixCache, SWATokenToKVPoolAllocator]:
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=1))
    req_to_token_pool = ReqToTokenPool(
        size=8, max_context_len=128, device="cpu", enable_memory_saver=False
    )
    kv_pool = SWAKVPool(
        size=KV_SIZE,
        size_swa=KV_SIZE,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=64,
        swa_attention_layer_ids=[0],
        full_attention_layer_ids=[1],
        device="cpu",
    )
    allocator = SWATokenToKVPoolAllocator(
        size=KV_SIZE,
        size_swa=KV_SIZE,
        page_size=1,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        sliding_window_size=WINDOW,
        tree_components=(FULL, SWA),
        tree_core_backend="python",
    )
    return UnifiedRadixCache(params), allocator


def _key(tokens: list[int]) -> RadixKey:
    return RadixKey(array("q", tokens))


def _free_full_ids(allocator: SWATokenToKVPoolAllocator) -> set[int]:
    full = allocator.full_attn_allocator
    ids = set(full.free_pages.tolist())
    release_pages = getattr(full, "release_pages", None)
    if release_pages is not None:
        ids |= set(release_pages.tolist())
    return ids


def _mapped_full_ids(cache: UnifiedRadixCache) -> dict[tuple[int, ...], list[int]]:
    """FULL ids per node key, for every node that holds a device value."""
    mapped: dict[tuple[int, ...], list[int]] = {}

    def walk(node):
        value = node.component_data[FULL].value
        if isinstance(value, torch.Tensor):
            mapped[tuple(node.key.token_ids)] = value.tolist()
        for child in node.children.values():
            walk(child)

    walk(cache.tree_core.root_node)
    return mapped


class TestSWARecoveryProtectedPrefix(CustomTestCase):
    def _insert_tombstoned_node(self, cache, allocator):
        """Tokens [1, 2, 3, 4] with FULL ids ``head``; the SWA side is evicted
        for the whole node, so it is a tombstone with a live FULL value."""
        head = allocator.alloc(4)
        result = cache.insert(
            InsertParams(
                key=_key([1, 2, 3, 4]),
                value=head,
                prev_prefix_len=0,
                component_evicted_seqlens={SWA: 4},
            )
        )
        node = cache.tree_core.node_by_id(result.last_device_node)
        self.assertIsNone(node.component_data[SWA].value)
        self.assertEqual(node.component_data[FULL].value.tolist(), head.tolist())
        return head, node

    def _reinsert_with_protected_prefix(self, cache, allocator, head):
        """A single-chunk request (SWA cursor 0) whose protected prefix [1, 2]
        borrows the node's ids by reference; only [3, 4] carries fresh KV."""
        fresh = allocator.alloc(2)
        row = torch.cat([head[:2], fresh])
        free_before = _free_full_ids(allocator)
        result = cache.insert(
            InsertParams(
                key=_key([1, 2, 3, 4]),
                value=row,
                prev_prefix_len=2,
                component_evicted_seqlens={SWA: 0},
            )
        )
        self.assertEqual(result.prefix_len, 4)
        freed = _free_full_ids(allocator) - free_before
        return fresh, freed

    def _assert_no_page_is_free_and_mapped(self, cache, freed):
        mapped = _mapped_full_ids(cache)
        still_mapped = {i for ids in mapped.values() for i in ids}
        self.assertEqual(sorted(freed & still_mapped), [], f"{mapped=} {freed=}")
        return mapped

    def test_unlocked_node_keeps_its_protected_head_and_recovers_the_tail(self):
        cache, allocator = _build_swa_cache()
        head, _ = self._insert_tombstoned_node(cache, allocator)
        fresh, freed = self._reinsert_with_protected_prefix(cache, allocator, head)

        mapped = self._assert_no_page_is_free_and_mapped(cache, freed)
        # The node splits at the protected prefix: the head keeps its canonical
        # ids, the tail adopts the fresh KV and frees only its own old ids.
        self.assertEqual(mapped[(1, 2)], head[:2].tolist())
        self.assertEqual(mapped[(3, 4)], fresh.tolist())
        self.assertEqual(sorted(freed), sorted(head[2:].tolist()))

    def test_locked_node_keeps_its_protected_head_and_frees_only_the_incoming_tail(
        self,
    ):
        cache, allocator = _build_swa_cache()
        head, node = self._insert_tombstoned_node(cache, allocator)
        # A matched request holds the node locked, so the recovery takes the
        # locked-FULL branch: the node keeps its ids and the incoming ones go.
        node.component_data[FULL].lock_ref += 1
        node.component_data[SWA].lock_ref += 1
        fresh, freed = self._reinsert_with_protected_prefix(cache, allocator, head)

        mapped = self._assert_no_page_is_free_and_mapped(cache, freed)
        self.assertEqual(mapped[(1, 2)], head[:2].tolist())
        self.assertEqual(mapped[(3, 4)], head[2:].tolist())
        self.assertEqual(sorted(freed), sorted(fresh.tolist()))


if __name__ == "__main__":
    unittest.main()
