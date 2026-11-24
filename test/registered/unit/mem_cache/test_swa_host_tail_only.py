"""SWA host backup under SGLANG_HICACHE_SWA_HOST_TAIL_ONLY: host DRAM keeps SWA
windows at leaves and existing branch points rather than copying every token's
SWA KV along unbranched pass-through chains."""

import unittest
from array import array
from collections import defaultdict
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

FULL, SWA = ComponentType.FULL, ComponentType.SWA
COMPONENTS = (FULL, SWA)
WINDOW = 4


class _Tree:
    """Device-resident nodes with SWA KV for unit-testing backup collection."""

    def __init__(self, *, tail_only=True, is_write_back=False):
        self.root = UnifiedTreeNode(COMPONENTS)
        self._next_id = 1
        self._skip_ids: set[int] = set()
        lru = SimpleNamespace(skips_host_backup=lambda n: n.id in self._skip_ids)
        self.core = SimpleNamespace(
            root_node=self.root,
            is_write_back=is_write_back,
            has_swa_host_pool=True,
            is_host_memory_buffer_only=False,
            lru_lists={SWA: lru},
            _planned_backup_node_ids=None,
        )
        self.comp = object.__new__(SWAComponent)
        self.comp.tree_core = self.core
        self.comp.component_type = SWA
        self.comp.sliding_window_size = WINDOW
        self.comp._host_tail_only = tail_only

    def add(
        self,
        tag,
        parent=None,
        *,
        seg=WINDOW,
        swa=True,
        host_swa=False,
        full_backed=False,
        skip_host_backup=False,
    ):
        parent = parent or self.root
        node = UnifiedTreeNode(COMPONENTS)
        node.id = self._next_id
        self._next_id += 1
        node.tag = tag
        node.key = list(range(seg))
        node.parent = parent
        node.component_data[FULL].value = torch.arange(seg)
        if full_backed:
            node.component_data[FULL].host_value = torch.arange(seg)
        if swa:
            node.component_data[SWA].value = torch.arange(seg)
        if host_swa:
            node.component_data[SWA].host_value = torch.arange(seg)
        if skip_host_backup:
            self._skip_ids.add(node.id)
        parent.children[tag] = node
        return node

    def collect(self, node, *, planned=()):
        self.core._planned_backup_node_ids = (
            frozenset(n.id for n in planned) if planned else None
        )
        try:
            return [n.tag for n in self.comp._collect_unbacked_swa_nodes(node)]
        finally:
            self.core._planned_backup_node_ids = None


def _build_real_core(*, tail_only: bool) -> UnifiedTreeCore:
    alloc = object.__new__(SWATokenToKVPoolAllocator)
    alloc.device = torch.device("cpu")
    cache = SimpleNamespace(
        token_to_kv_pool_allocator=alloc,
        page_size=1,
        enable_session_radix_cache=False,
    )
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=alloc,
        page_size=1,
        sliding_window_size=WINDOW,
        tree_components=COMPONENTS,
    )
    full = FullComponent(cache, params)
    swa = SWAComponent(cache, params)
    swa._host_tail_only = tail_only
    core = UnifiedTreeCore(params, {FULL: full, SWA: swa})
    core.set_hicache_enabled()
    core.has_swa_host_pool = True
    return core


def _demote(core: UnifiedTreeCore, node_id: int) -> None:
    res = core.demote(node_id)
    res.device_frees.clear()
    res.host_frees.clear()


class TestSwaHostTailOnly(unittest.TestCase):
    def test_older_part_of_a_capped_leaf_is_not_copied(self):
        # Write-through backs up [head, tail] in one BackupKV batch: tail is in
        # the planned batch, so head skips SWA copy and only tail's window copies.
        t = _Tree()
        head = t.add("head", seg=5 * WINDOW)
        tail = t.add("tail", head)
        self.assertEqual(t.collect(head, planned=(head, tail)), [])
        self.assertEqual(t.collect(tail, planned=(head, tail)), ["tail"])

        # Same skip holds once tail already has its host SWA copy.
        t2 = _Tree()
        head2 = t2.add("head", seg=5 * WINDOW)
        t2.add("tail", head2, host_swa=True)
        self.assertEqual(t2.collect(head2), [])

    def test_device_only_child_not_reaching_host_does_not_shadow_parent(self):
        # Write-through selective: head backs up alone while tail stays device-only
        # (not in the BackupKV batch and no host copy) -> head keeps its window.
        t = _Tree(is_write_back=False)
        head = t.add("head", seg=5 * WINDOW)
        t.add("tail", head)
        self.assertEqual(t.collect(head, planned=(head,)), ["head"])

        # Write-back: unbacked non-stale device tail WILL back up on eviction,
        # so head skips; but if tail already has Full on host (demotes without
        # SWA backup) or skips_host_backup is set, head keeps its window.
        wb_ok = _Tree(is_write_back=True)
        h_ok = wb_ok.add("head", seg=5 * WINDOW)
        wb_ok.add("tail", h_ok)
        self.assertEqual(wb_ok.collect(h_ok), [])

        wb_full_backed = _Tree(is_write_back=True)
        h_fb = wb_full_backed.add("head", seg=5 * WINDOW)
        wb_full_backed.add("tail", h_fb, full_backed=True)
        self.assertEqual(wb_full_backed.collect(h_fb), ["head"])

        wb_stale = _Tree(is_write_back=True)
        h_st = wb_stale.add("head", seg=5 * WINDOW)
        wb_stale.add("tail", h_st, skip_host_backup=True)
        self.assertEqual(wb_stale.collect(h_st), ["head"])

    def test_a_short_child_still_needs_the_parent_window(self):
        t = _Tree()
        head = t.add("head", seg=5 * WINDOW)
        short = t.add("short", head, seg=WINDOW - 1)
        self.assertEqual(t.collect(head, planned=(head, short)), ["head"])
        self.assertEqual(t.collect(short, planned=(head, short)), ["short", "head"])

    def test_the_child_must_hold_a_window(self):
        t = _Tree()
        head = t.add("head", seg=5 * WINDOW)
        dropped = t.add("dropped", head, swa=False)
        self.assertEqual(t.collect(head, planned=(head, dropped)), ["head"])

    def test_unbranched_first_divergence_vs_branch_point(self):
        # Single-child pass-through skips host SWA until a second branch exists;
        # once branched, backing up the shared prefix retains its window.
        t = _Tree()
        shared = t.add("shared", seg=5 * WINDOW)
        a = t.add("session_a", shared, host_swa=True)
        self.assertEqual(t.collect(shared, planned=(shared, a)), [])
        t.add("session_b", shared, host_swa=True)
        self.assertEqual(t.collect(shared), ["shared"])

    def test_evicting_shadowing_host_leaf_reclaims_windowless_parent(self):
        # When host leaf eviction drops tail, a pass-through head with host Full
        # KV and no SWA window cannot match any prompt and must be reclaimed with
        # tail instead of stranding dead Full KV in the host arena.
        core = _build_real_core(tail_only=True)
        head = core._add_new_node(
            core.root_node,
            RadixKey(array("q", range(0, 5 * WINDOW)), None),
            torch.arange(0, 5 * WINDOW),
        )
        tail = core._add_new_node(
            head,
            RadixKey(array("q", range(5 * WINDOW, 6 * WINDOW)), None),
            torch.arange(5 * WINDOW, 6 * WINDOW),
        )
        head.component_data[FULL].host_value = torch.arange(0, 5 * WINDOW)
        tail.component_data[FULL].host_value = torch.arange(5 * WINDOW, 6 * WINDOW)
        tail.component_data[SWA].host_value = torch.arange(5 * WINDOW, 6 * WINDOW)
        tail.component_data[SWA].value = torch.arange(5 * WINDOW, 6 * WINDOW)
        _demote(core, tail.id)
        _demote(core, head.id)
        self.assertIn(tail, core.evictable_host_leaves)

        tracker = defaultdict(int)
        dev_frees = defaultdict(list)
        host_frees = defaultdict(list)
        core._evict_host_leaf(tail, tracker, dev_frees, host_frees)
        self.assertEqual(tracker[FULL], 6 * WINDOW)
        self.assertEqual(tracker[SWA], WINDOW)
        self.assertEqual(len(core.root_node.children), 0)

        # Conversely, a parent that DOES hold its own host SWA window (e.g. an
        # earlier standalone turn) survives when its child leaf is host-evicted.
        turn1 = core._add_new_node(
            core.root_node,
            RadixKey(array("q", range(100, 100 + 2 * WINDOW)), None),
            torch.arange(0, 2 * WINDOW),
        )
        turn2 = core._add_new_node(
            turn1,
            RadixKey(array("q", range(200, 200 + WINDOW)), None),
            torch.arange(2 * WINDOW, 3 * WINDOW),
        )
        turn1.component_data[FULL].host_value = torch.arange(0, 2 * WINDOW)
        turn1.component_data[SWA].host_value = torch.arange(WINDOW, 2 * WINDOW)
        turn1.component_data[SWA].value = torch.arange(WINDOW, 2 * WINDOW)
        turn2.component_data[FULL].host_value = torch.arange(2 * WINDOW, 3 * WINDOW)
        turn2.component_data[SWA].host_value = torch.arange(2 * WINDOW, 3 * WINDOW)
        turn2.component_data[SWA].value = torch.arange(2 * WINDOW, 3 * WINDOW)
        _demote(core, turn2.id)
        _demote(core, turn1.id)

        tracker2 = defaultdict(int)
        core._evict_host_leaf(turn2, tracker2, defaultdict(list), defaultdict(list))
        self.assertEqual(tracker2[FULL], WINDOW)
        self.assertIn(turn1, core.evictable_host_leaves)

    def test_off_by_default_copies_every_node(self):
        t = _Tree(tail_only=False)
        head = t.add("head", seg=5 * WINDOW)
        tail = t.add("tail", head)
        self.assertEqual(t.collect(head, planned=(head, tail)), ["head"])


if __name__ == "__main__":
    unittest.main()
