"""SWA host backup under SGLANG_HICACHE_SWA_HOST_TAIL_ONLY: a node's SWA KV is copied to
host only when some match can end there and need it. Copying every token's SWA KV made
the shared unified host arena drop the oldest windows, so host-resident Full KV could
no longer be matched and cached prefixes were recomputed."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

FULL, SWA = ComponentType.FULL, ComponentType.SWA
COMPONENTS = (FULL, SWA)
WINDOW = 4


class _Tree:
    """Device-resident nodes with SWA KV and no host copies yet."""

    def __init__(self, *, tail_only=True):
        self.root = UnifiedTreeNode(COMPONENTS)
        core = SimpleNamespace(
            root_node=self.root,
            is_write_back=False,
            has_swa_host_pool=True,
            is_host_memory_buffer_only=False,
        )
        self.comp = object.__new__(SWAComponent)
        self.comp.tree_core = core
        self.comp.component_type = SWA
        self.comp.sliding_window_size = WINDOW
        self.comp._host_tail_only = tail_only

    def add(self, tag, parent=None, *, seg=WINDOW, swa=True):
        parent = parent or self.root
        node = UnifiedTreeNode(COMPONENTS)
        node.tag = tag
        node.key = list(range(seg))
        node.parent = parent
        node.component_data[FULL].value = torch.arange(seg)
        if swa:
            node.component_data[SWA].value = torch.arange(seg)
        parent.children[tag] = node
        return node

    def collect(self, node):
        return [n.tag for n in self.comp._collect_unbacked_swa_nodes(node)]


class TestSwaHostTailOnly(unittest.TestCase):
    def test_older_part_of_a_capped_leaf_is_not_copied(self):
        # Insert splits a long leaf into head + one-window tail: only the tail's
        # SWA KV is ever read, so only it is copied.
        t = _Tree()
        head = t.add("head", seg=5 * WINDOW)
        tail = t.add("tail", head)
        self.assertEqual(t.collect(head), [])
        self.assertEqual(t.collect(tail), ["tail"])

    def test_a_short_child_still_needs_the_parent_window(self):
        # The child's segment is shorter than the window, so a match ending at the
        # child reads into the parent: both are copied.
        t = _Tree()
        head = t.add("head", seg=5 * WINDOW)
        short = t.add("short", head, seg=WINDOW - 1)
        self.assertEqual(t.collect(head), ["head"])
        self.assertEqual(t.collect(short), ["short", "head"])

    def test_the_child_must_hold_a_window(self):
        # The child lost its SWA KV (a tombstoned prefill chunk), so a match ending
        # at the child needs the parent's window.
        t = _Tree()
        head = t.add("head", seg=5 * WINDOW)
        t.add("dropped", head, swa=False)
        self.assertEqual(t.collect(head), ["head"])

    def test_a_branch_point_keeps_its_window(self):
        # A shared prefix (system prompt) with two sessions below it: a third
        # session that shares only the prefix ends its match here.
        t = _Tree()
        shared = t.add("shared", seg=5 * WINDOW)
        t.add("session_a", shared)
        t.add("session_b", shared)
        self.assertEqual(t.collect(shared), ["shared"])

    def test_off_by_default_copies_every_node(self):
        t = _Tree(tail_only=False)
        head = t.add("head", seg=5 * WINDOW)
        t.add("tail", head)
        self.assertEqual(t.collect(head), ["head"])


if __name__ == "__main__":
    unittest.main()
