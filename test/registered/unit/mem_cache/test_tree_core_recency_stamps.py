"""CPU checks of the Python TreeCore's wall-clock recency stamps."""

import unittest
from array import array
from unittest import mock

import test_unified_radix_cache_unittest as radix_fixture
from test_unified_radix_cache_unittest import CacheConfig, build_fixture

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestTreeCoreRecencyStamps(CustomTestCase):
    def setUp(self):
        cfg = CacheConfig(
            components=(ComponentType.FULL, ComponentType.SWA),
            sliding_window_size=4,
        )
        # The fixture asks for an accelerator; the CPU CI runner has none.
        with mock.patch.object(radix_fixture, "get_device", return_value="cpu"):
            self.cache, self.allocator, _ = build_fixture(
                cfg, tree_core_backend="python"
            )
        self.core = self.cache.tree_core

    def _insert(self, tokens):
        value = self.allocator.alloc(len(tokens))
        self.cache.insert(
            InsertParams(key=RadixKey(array("q", tokens)), value=value[: len(tokens)])
        )

    def _node(self, tokens):
        result = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        )
        return self.core.node_by_id(result.last_device_node)

    def test_access_and_split_stamp_both_halves(self):
        """A match that diverges inside a node refreshes the new parent and the
        child, so both carry the match's wall time, not the insert's."""
        with mock.patch("time.monotonic", return_value=100.0):
            self._insert(list(range(1, 9)))
        with mock.patch("time.monotonic", return_value=200.0):
            self._node([1, 2, 3, 4, 5, 6, 99])
        # The leaf below the divergence is off the match path; only the split
        # stamps it.
        node = self.core.root_node
        while node.children:
            (node,) = node.children.values()
        self.assertEqual(
            (node.parent.last_access_wall, node.last_access_wall), (200.0, 200.0)
        )

    def test_split_prefix_keeps_the_last_use_of_slots_it_does_not_reinsert(self):
        """A split prefix whose SWA is not on device enters no device LRU, so its
        device slot must keep the child's last use rather than the split time."""
        seq = list(range(1, 9))
        with mock.patch("time.monotonic", return_value=100.0):
            value = self.allocator.alloc(len(seq))
            self.cache.insert(
                InsertParams(
                    key=RadixKey(array("q", seq)),
                    value=value[: len(seq)],
                    component_evicted_seqlens={ComponentType.SWA: len(seq)},
                )
            )
        with mock.patch("time.monotonic", return_value=200.0):
            self._node([1, 2, 3, 4, 5, 6, 99])
        slot = self.core.lru_lists[ComponentType.SWA]._pt
        (prefix,) = self.core.root_node.children.values()
        self.assertEqual(prefix.lru_wall[slot], 100.0)

    def test_lru_slot_stamps_follow_mru_moves_but_not_demotion(self):
        """Moving a node to the MRU end stamps that LRU slot; demoting it to the
        eviction end, or repositioning it for a session-coverage change, is not a
        use and keeps the stamp."""
        lru = self.core.lru_lists[ComponentType.SWA]
        with mock.patch("time.monotonic", return_value=100.0):
            self._insert(list(range(1, 9)))
        node = self._node(list(range(1, 9)))
        with mock.patch("time.monotonic", return_value=300.0):
            lru.reset_node_mru(node)
        with mock.patch("time.monotonic", return_value=400.0):
            lru.reset_node_lru(node)
            self.core.components_by_type[ComponentType.SWA]._refresh_session_partition(
                node
            )
        self.assertEqual(node.lru_wall[lru._pt], 300.0)


if __name__ == "__main__":
    unittest.main()
