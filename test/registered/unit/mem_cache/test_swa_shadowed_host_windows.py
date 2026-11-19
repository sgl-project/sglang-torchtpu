"""SWA host reclaim on a shared Full/SWA host arena: a host window is
reclaimed early only on a pass-through node, whose one child holds Full KV and a window
of its own. Reclaiming every window with any deeper boundary below it emptied whole
chunked-prefill paths of windows, so losing the last one recomputed the entire prefix.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedLRUList,
    UnifiedTreeNode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

FULL, SWA = ComponentType.FULL, ComponentType.SWA
COMPONENTS = (FULL, SWA)
WINDOW = 4


class _Tree:
    """Host-only nodes (Full and SWA on host); add() order is LRU order."""

    def __init__(self):
        self.lru = UnifiedLRUList(SWA, COMPONENTS, use_host_ptr=True)
        self.root = UnifiedTreeNode(COMPONENTS)
        self.tombstoned = []

        def tombstone(node, comp, target, tracker, device_frees, host_frees):
            self.tombstoned.append(node.tag)
            tracker[SWA] += len(node.component_data[SWA].host_value)
            node.component_data[SWA].host_value = None
            self.lru.remove_node(node)

        self.core = SimpleNamespace(
            host_lru_lists={SWA: self.lru},
            evictable_host_leaves=set(),
            _evict_component_and_detach_lru=tombstone,
            _cascade_evict=lambda *args, **kwargs: None,
        )
        self.comp = object.__new__(SWAComponent)
        self.comp.tree_core = self.core
        self.comp.sliding_window_size = WINDOW

    def add(self, tag, parent=None, *, seg=WINDOW, window=True):
        node = UnifiedTreeNode(COMPONENTS)
        node.tag = tag
        node.key = list(range(seg))
        node.component_data[FULL].host_value = torch.arange(seg)
        if window:
            node.component_data[SWA].host_value = torch.arange(WINDOW)
            self.lru.insert_mru(node)
        if parent is not None:
            parent.children[tag] = node
            node.parent = parent
        return node

    def leaf(self, node):
        self.core.evictable_host_leaves.add(node)

    def reclaim(self, num_tokens):
        tracker = {SWA: 0}
        self.comp.evict_shadowed_host_windows(num_tokens, tracker, {}, {})
        return tracker[SWA]


class TestSwaShadowedHostWindows(unittest.TestCase):
    def test_only_shadowed_internal_windows_go(self):
        # a -> b -> c(leaf): a and b are shadowed by c. d -> e(leaf, no window): d is
        # e's only boundary and must stay even though it is the oldest.
        t = _Tree()
        d = t.add("d")
        a = t.add("a")
        b = t.add("b", a)
        t.leaf(t.add("c", b))
        t.leaf(t.add("e", d, window=False))
        self.assertEqual(t.reclaim(10 * WINDOW), 2 * WINDOW)
        self.assertEqual(t.tombstoned, ["a", "b"])

    def test_a_short_child_segment_does_not_shadow(self):
        # The child's own segment is shorter than the window, so its boundary still
        # reads the parent's window; the parent is not shadowed.
        t = _Tree()
        p = t.add("p")
        t.leaf(t.add("q", p, seg=WINDOW - 1))
        self.assertEqual(t.reclaim(WINDOW), 0)
        self.assertEqual(t.tombstoned, [])

    def test_a_branch_point_keeps_its_window(self):
        # Two sessions below a shared prefix: a third one that shares only the prefix
        # ends its match here.
        t = _Tree()
        p = t.add("p")
        t.leaf(t.add("x", p))
        t.leaf(t.add("y", p))
        self.assertEqual(t.reclaim(WINDOW), 0)

    def test_chunk_end_windows_survive(self):
        # Chunked prefill: [w1] -> [chunk part with no SWA] -> [w2]. A deeper window
        # exists below w1, but w1 is the fallback when w2 goes; it must stay.
        t = _Tree()
        w1 = t.add("w1")
        part = t.add("part", w1, seg=5 * WINDOW, window=False)
        t.leaf(t.add("w2", part))
        self.assertEqual(t.reclaim(10 * WINDOW), 0)
        self.assertEqual(t.tombstoned, [])

    def test_usable_window_spans_short_nodes(self):
        t = _Tree()
        t.core.root_node = t.root
        a = t.add("a", t.root, seg=WINDOW - 1)
        b = t.add("b", a, seg=1)
        gap = t.add("gap", b, seg=1, window=False)
        self.assertTrue(t.comp.has_usable_window(b))
        self.assertFalse(t.comp.has_usable_window(gap))


if __name__ == "__main__":
    unittest.main()
