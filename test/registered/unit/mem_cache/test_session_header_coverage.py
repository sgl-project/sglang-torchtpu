"""Session coverage under header-derived session ids (one id per conversation, never closed):
helper calls, context edits, growth without closes, and the SWA host partition."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
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


def make_cache(size=512):
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
            tree_components=(FULL, SWA),
        )
    )


def finish(cache, tokens, session_id):
    """A finished request of this session: insert its KV and register the leaf."""
    indices = cache.token_to_kv_pool_allocator.alloc(len(tokens))
    cache.insert(
        InsertParams(key=RadixKey(array("q", tokens)), value=indices.to(torch.int64))
    )
    leaf = cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", tokens)))
    ).last_device_node
    cache.session_refs.register_session_ref(
        SimpleNamespace(
            session_id=session_id,
            session_generation=cache.ensure_session_generation(session_id),
            session=None,
            origin_input_ids=array("q", tokens),
            output_ids=array("q"),
            extra_key=None,
        ),
        leaf=leaf,
    )
    return cache.tree_core.node_by_id(leaf)


def full_ref(cache, node):
    return cache.components[FULL].session_ref(node)


def swa_covered(cache, session_id):
    """SWA tokens the session's leaves cover. Coverage is node-granular: the walk from a
    leaf takes whole nodes until it reaches window + one page, so a long last node
    (a whole turn's new tokens, up to the insert stride) is covered in full."""
    swa = cache.components[SWA]
    return sum(swa._session_leaf_covered_len.get(session_id, {}).values())


def leaves(cache, ct, session_id):
    return cache.components[ct]._session_leaves.get(session_id, set())


SYSTEM = list(range(1, 17))  # the conversation's system prompt
HELPER_SYSTEM = list(range(901, 909))  # a short helper call's own system prompt


class TestHelperCallsAndEdits(CustomTestCase):
    def setUp(self):
        self.cache = make_cache()

    def test_helper_call_adds_a_leaf_and_keeps_the_main_coverage(self):
        main1 = finish(self.cache, SYSTEM + [20, 21, 22, 23], "conv")
        helper = finish(self.cache, HELPER_SYSTEM + [30, 31], "conv")
        # The helper call shares no prefix with the conversation: a second leaf.
        self.assertEqual(leaves(self.cache, FULL, "conv"), {main1, helper})
        self.assertEqual(full_ref(self.cache, main1), 1)
        self.assertEqual(full_ref(self.cache, helper), 1)
        # Whole nodes: the 20-token first turn and the 10-token helper call.
        self.assertEqual(swa_covered(self.cache, "conv"), 20 + 10)

        # The next main turn extends the main branch: its coverage moves forward,
        # the helper's leaf is untouched.
        main2 = finish(self.cache, SYSTEM + [20, 21, 22, 23, 24, 25, 26], "conv")
        self.assertEqual(leaves(self.cache, FULL, "conv"), {main2, helper})
        self.assertEqual(full_ref(self.cache, main2), 1)
        self.assertEqual(full_ref(self.cache, helper), 1)
        # Extending the branch split the first turn at the window: the walk takes the
        # 3-token new node and the 4-token node before it (>= window + page), plus the helper.
        self.assertEqual(swa_covered(self.cache, "conv"), 3 + 4 + 10)
        self.cache.sanity_check()

    def test_context_edit_leaves_the_old_branch_covered(self):
        before = finish(self.cache, SYSTEM + [20, 21, 22, 23, 24, 25], "conv")
        # The trace edits earlier context: the follow-up diverges inside the last turn.
        after = finish(self.cache, SYSTEM + [20, 21, 40, 41, 42, 43, 44], "conv")
        self.assertEqual(leaves(self.cache, FULL, "conv"), {before, after})
        # Both branches stay covered until the session closes, which AgentX never does.
        self.assertEqual(full_ref(self.cache, before), 1)
        self.assertEqual(full_ref(self.cache, after), 1)
        self.cache.release_radix_session("conv")
        self.assertEqual(full_ref(self.cache, before), 0)
        self.assertEqual(full_ref(self.cache, after), 0)
        self.cache.sanity_check()


class TestGrowthWithoutCloses(CustomTestCase):
    def test_bookkeeping_and_covered_windows_grow_with_session_ids(self):
        cache = make_cache(size=2048)
        refs = cache.session_refs
        logger = "sglang.srt.mem_cache.unified_cache.session_ref_tracker"
        with self.assertLogs(logger, level="INFO") as logs:
            for i in range(40):
                finish(cache, [1000 + 50 * i + k for k in range(12)], f"conv-{i}")
        # Logged at the first id (then every 100th): ids, leaves, covered SWA tokens.
        self.assertEqual(len(logs.output), 1)
        self.assertIn("Radix sessions: 1 open ids", logs.output[0])
        # One generation, one leaf set per component and one SWA coverage record per
        # id; nothing shrinks without /close_session.
        self.assertEqual(len(refs._session_generations), 40)
        self.assertEqual(len(cache.components[FULL]._session_leaves), 40)
        self.assertEqual(len(cache.components[SWA]._session_leaves), 40)
        self.assertEqual(len(cache.components[SWA]._session_leaf_covered_len), 40)
        # Each session covers its whole 12-token node (node-granular, > window + page).
        self.assertEqual(
            sum(swa_covered(cache, f"conv-{i}") for i in range(40)), 40 * 12
        )
        # Advancing a session does not add bookkeeping.
        finish(cache, [1000 + k for k in range(12)] + [7, 8, 9], "conv-0")
        self.assertEqual(len(refs._session_generations), 40)
        self.assertEqual(swa_covered(cache, "conv-0"), 3 + 4)
        cache.sanity_check()


class TestSWAHostPartition(CustomTestCase):
    """Write-through backs every window up to the SWA host pool; its LRU keeps covered
    windows in the referenced partition, so host eviction takes uncovered ones first."""

    def build(self, covered):
        is_ref = lambda n: n.component_data[SWA].session_ref > 0
        host_lru = UnifiedLRUList(
            SWA, (FULL, SWA), use_host_ptr=True, is_referenced=is_ref
        )
        evicted = []

        def evict(node, comp, target, tracker, device_frees, host_frees):
            evicted.append(node)
            tracker[SWA] += len(node.component_data[SWA].host_value)
            node.component_data[SWA].host_value = None
            host_lru.remove_node(node)

        core = SimpleNamespace(
            host_lru_lists={SWA: host_lru},
            lru_lists={SWA: UnifiedLRUList(SWA, (FULL, SWA), is_referenced=is_ref)},
            enable_session_radix_cache=True,
            swa_host_evict_passed_first=False,
            evictable_host_leaves=set(),
            _evict_component_and_detach_lru=evict,
            _cascade_evict=lambda *args, **kwargs: None,
        )
        comp = object.__new__(SWAComponent)
        comp.tree_core = core
        nodes = []
        for i in range(len(covered)):
            node = UnifiedTreeNode((FULL, SWA))
            node.component_data[SWA].host_value = torch.arange(WINDOW)
            node.component_data[SWA].session_ref = int(covered[i])
            host_lru.insert_mru(node)
            nodes.append(node)
        return comp, nodes, evicted

    def drive(self, comp, num_tokens):
        tracker = {SWA: 0}
        comp.drive_host_eviction(num_tokens, tracker, {}, {})
        return tracker[SWA]

    def test_uncovered_windows_go_first_then_covered_in_lru_order(self):
        comp, nodes, evicted = self.build([True, False, True, False])
        self.assertEqual(self.drive(comp, 3 * WINDOW), 3 * WINDOW)
        self.assertEqual(evicted, [nodes[1], nodes[3], nodes[0]])

    def test_coverage_change_moves_a_window_between_partitions(self):
        comp, nodes, evicted = self.build([False, False, False])
        # The session advances onto nodes[0]: its window joins the covered partition.
        nodes[0].component_data[SWA].session_ref = 1
        comp._refresh_session_partition(nodes[0])
        self.assertEqual(self.drive(comp, 2 * WINDOW), 2 * WINDOW)
        self.assertEqual(evicted, [nodes[1], nodes[2]])

    def test_passed_first_is_off_with_session_radix_cache(self):
        with envs.SGLANG_HICACHE_SWA_HOST_EVICT_PASSED_FIRST.override(True):
            cache = make_cache()
        self.assertFalse(cache.tree_core.swa_host_evict_passed_first)


if __name__ == "__main__":
    unittest.main()
