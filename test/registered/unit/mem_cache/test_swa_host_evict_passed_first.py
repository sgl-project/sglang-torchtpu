"""SWA host eviction with SGLANG_HICACHE_SWA_HOST_EVICT_PASSED_FIRST: windows a match
walked past go first, in LRU order, before any other host window."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedLRUList,
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

SWA = ComponentType.SWA
COMPONENTS = (ComponentType.FULL, ComponentType.SWA)
WINDOW = 4


def build(passed_first: bool, num_nodes: int):
    """Host-only SWA windows of WINDOW tokens; nodes[0] is the LRU end."""
    lru = UnifiedLRUList(SWA, COMPONENTS, use_host_ptr=True)
    evicted = []

    def evict(node, comp, target, tracker, device_frees, host_frees):
        evicted.append(node)
        tracker[SWA] += len(node.component_data[SWA].host_value)
        node.component_data[SWA].host_value = None
        lru.remove_node(node)

    core = SimpleNamespace(
        host_lru_lists={SWA: lru},
        enable_session_radix_cache=False,
        swa_host_evict_passed_first=passed_first,
        evictable_host_leaves=set(),
        _evict_component_and_detach_lru=evict,
        _cascade_evict=lambda *args, **kwargs: None,
    )
    comp = object.__new__(SWAComponent)
    comp.tree_core = core
    nodes = []
    for _ in range(num_nodes):
        node = UnifiedTreeNode(COMPONENTS)
        node.component_data[SWA].host_value = torch.arange(WINDOW)
        lru.insert_mru(node)
        nodes.append(node)
    return comp, nodes, evicted


def drive(comp, num_tokens):
    tracker = {SWA: 0}
    comp.drive_host_eviction(num_tokens, tracker, {}, {})
    return tracker[SWA]


def pass_node(node):
    UnifiedTreeCore.mark_match_passed(SimpleNamespace(node_by_id=lambda _: node), 0)


class TestSWAHostEvictPassedFirst(unittest.TestCase):
    def test_off_is_plain_lru(self):
        comp, nodes, evicted = build(False, 4)
        pass_node(nodes[3])
        self.assertEqual(drive(comp, 2 * WINDOW), 2 * WINDOW)
        self.assertEqual(evicted, nodes[:2])

    def test_passed_windows_go_first_in_lru_order(self):
        comp, nodes, evicted = build(True, 5)
        pass_node(nodes[3])
        pass_node(nodes[1])
        self.assertEqual(drive(comp, 3 * WINDOW), 3 * WINDOW)
        # The two passed windows (LRU order), then the LRU end.
        self.assertEqual(evicted, [nodes[1], nodes[3], nodes[0]])

    def test_a_later_match_ending_here_clears_passed(self):
        comp, nodes, evicted = build(True, 3)
        pass_node(nodes[2])
        nodes[2].match_end_time = nodes[2].match_pass_time + 1
        self.assertEqual(drive(comp, WINDOW), WINDOW)
        self.assertEqual(evicted, [nodes[0]])

    def test_stops_once_enough_is_freed(self):
        comp, nodes, evicted = build(True, 4)
        for node in nodes:
            pass_node(node)
        self.assertEqual(drive(comp, WINDOW), WINDOW)
        self.assertEqual(evicted, [nodes[0]])


if __name__ == "__main__":
    unittest.main()
