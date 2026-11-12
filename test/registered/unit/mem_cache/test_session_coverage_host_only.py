"""--session-coverage-host-only: session coverage orders the host LRU lists only; device
eviction (Full and SWA) stays plain LRU over the hot working set."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedLRUList,
    UnifiedTreeNode,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.test_utils import CustomTestCase

FULL, SWA = ComponentType.FULL, ComponentType.SWA
WINDOW = 4
LAYERS, FULL_LAYERS = 4, (3,)


def make_cache(host_only, size=512):
    kv_pool = SWAKVPool(
        size=size,
        size_swa=size,
        page_size=1,
        dtype=torch.float16,
        head_num=1,
        head_dim=4,
        swa_attention_layer_ids=[i for i in range(LAYERS) if i not in FULL_LAYERS],
        full_attention_layer_ids=list(FULL_LAYERS),
        device="cpu",
    )
    allocator = SWATokenToKVPoolAllocator(
        size=size,
        size_swa=size,
        page_size=1,
        dtype=torch.float16,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    req_pool = ReqToTokenPool(
        size=8, max_context_len=256, device="cpu", enable_memory_saver=False
    )
    return UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            sliding_window_size=WINDOW,
            eviction_policy="lru",
            enable_session_radix_cache=True,
            session_leaf_at_prompt_end=True,
            session_coverage_host_only=host_only,
            tree_components=(FULL, SWA),
        )
    )


def insert(cache, tokens):
    indices = cache.token_to_kv_pool_allocator.alloc(len(tokens))
    node_id = cache.insert(
        InsertParams(key=RadixKey(array("q", tokens)), value=indices.to(torch.int64))
    ).last_device_node
    return cache.tree_core.node_by_id(node_id)


def finish_turn(cache, prompt, output, session_id):
    """End-of-prefill insert, finished insert, prompt-end session leaf."""
    insert(cache, prompt)
    last = insert(cache, prompt + output)
    leaf = cache.finished_session_leaf(last.id, len(prompt))
    cache.session_refs.register_session_ref(
        SimpleNamespace(
            session_id=session_id,
            session_generation=cache.ensure_session_generation(session_id),
            session=None,
            origin_input_ids=array("q", prompt),
            output_ids=array("q", output),
            extra_key=None,
        ),
        leaf=leaf,
    )
    return cache.tree_core.node_by_id(leaf)


IDLE = [100 + k for k in range(12)]  # an idle session's prompt (covered at its end)
ACTIVE = [
    500 + k for k in range(12)
]  # an active request's prompt, used later (uncovered)


def setup_idle_then_active(host_only):
    cache = make_cache(host_only)
    idle = finish_turn(cache, IDLE, [190, 191], "idle")
    # The idle session's output branch is uncovered and goes first in both modes.
    cache.evict(EvictParams(num_tokens=2, swa_num_tokens=2))
    active = insert(cache, ACTIVE)
    assert cache.components[SWA].session_ref(idle) == 1
    assert cache.components[SWA].session_ref(active) == 0
    return cache, idle, active


def has_swa(node):
    return node.component_data[SWA].value is not None


def has_full(cache, node):
    return node.id in cache.tree_core._node_arena and (
        node.component_data[FULL].value is not None
    )


class TestDeviceOrder(CustomTestCase):
    def test_default_keeps_the_covered_window_on_device(self):
        cache, idle, active = setup_idle_then_active(host_only=False)
        # One prompt's SWA: its out-of-window part and its last window (split nodes).
        cache.evict(EvictParams(num_tokens=0, swa_num_tokens=len(ACTIVE)))
        # Covered windows go last on device: the active request loses its window.
        self.assertTrue(has_swa(idle))
        self.assertFalse(has_swa(active))
        cache.sanity_check()

    def test_host_only_keeps_the_active_window_on_device(self):
        cache, idle, active = setup_idle_then_active(host_only=True)
        # One prompt's SWA: its out-of-window part and its last window (split nodes).
        cache.evict(EvictParams(num_tokens=0, swa_num_tokens=len(ACTIVE)))
        # Plain LRU on device: the idle session's older window goes first.
        self.assertFalse(has_swa(idle))
        self.assertTrue(has_swa(active))
        cache.sanity_check()

    def test_host_only_full_device_eviction_is_plain_lru(self):
        for host_only, idle_kept in ((False, True), (True, False)):
            with self.subTest(host_only=host_only):
                cache, idle, active = setup_idle_then_active(host_only)
                cache.evict(EvictParams(num_tokens=len(ACTIVE)))
                self.assertEqual(has_full(cache, idle), idle_kept)
                self.assertEqual(has_full(cache, active), not idle_kept)
                cache.sanity_check()

    def test_lists(self):
        core = make_cache(host_only=True).tree_core
        # Device lists unpartitioned, host lists partitioned by coverage.
        self.assertIsNone(core.lru_lists[SWA].mid)
        self.assertIsNotNone(core.host_lru_lists[SWA].mid)
        default = make_cache(host_only=False).tree_core
        self.assertIsNotNone(default.lru_lists[SWA].mid)


class TestHostOrder(CustomTestCase):
    """On host the reverse holds: an idle session's covered window outlives a newer
    uncovered one, through the same partitioned host list the flag keeps."""

    def test_covered_window_outlives_a_newer_uncovered_one_on_host(self):
        core = make_cache(host_only=True).tree_core
        is_ref = core._session_lru_predicate(SWA, host=True)
        self.assertIsNotNone(is_ref)
        self.assertIsNone(core._session_lru_predicate(SWA, host=False))
        host_lru = UnifiedLRUList(
            SWA, (FULL, SWA), use_host_ptr=True, is_referenced=is_ref
        )
        evicted = []

        def evict(node, comp, target, tracker, device_frees, host_frees):
            evicted.append(node)
            tracker[SWA] += len(node.component_data[SWA].host_value)
            node.component_data[SWA].host_value = None
            host_lru.remove_node(node)

        ns = SimpleNamespace(
            host_lru_lists={SWA: host_lru},
            enable_session_radix_cache=True,
            swa_host_evict_passed_first=False,
            evictable_host_leaves=set(),
            _evict_component_and_detach_lru=evict,
            _cascade_evict=lambda *args, **kwargs: None,
        )
        comp = object.__new__(SWAComponent)
        comp.tree_core = ns
        idle, active = UnifiedTreeNode((FULL, SWA)), UnifiedTreeNode((FULL, SWA))
        for node, ref in ((idle, 1), (active, 0)):  # idle is older
            node.component_data[SWA].host_value = torch.arange(WINDOW)
            node.component_data[SWA].session_ref = ref
            host_lru.insert_mru(node)
        tracker = {SWA: 0}
        comp.drive_host_eviction(WINDOW, tracker, {}, {})
        self.assertEqual(evicted, [active])


if __name__ == "__main__":
    unittest.main()
