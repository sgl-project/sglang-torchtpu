"""CPU checks of the decode side of the P/D prefix pull on a real full + SWA tree."""

import pkgutil
import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

import test_unified_radix_cache_unittest as radix_fixture
from test_unified_radix_cache_unittest import CacheConfig, build_fixture

from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.prefix_pull import pullable_by_room
from sglang_tpu.disaggregation.prefix_pull_decode import (
    PromptEndAnchors,
    PullablePrefix,
    PullHolds,
    decode_pullable,
    prompt_end_anchor_node,
)
from sglang_tpu.envs import envs
from sglang_tpu.hooks import HOOKS, pd_prefix_pull

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SWA = ComponentType.SWA


class _Req(SimpleNamespace):
    """Weak-referenceable, as the metrics hook keys requests weakly."""

    __hash__ = object.__hash__


def _req(prompt, rid="r", last_node=None, room=None, host="10.0.0.1"):
    return _Req(
        bootstrap_host=host,
        origin_input_ids=list(prompt),
        extra_key=None,
        cache_salt=None,
        rid=rid,
        last_node=last_node,
        bootstrap_room=room,
    )


class _Base(CustomTestCase):
    def setUp(self):
        cfg = CacheConfig(components=(ComponentType.FULL, SWA), sliding_window_size=4)
        with mock.patch.object(radix_fixture, "get_device", return_value="cpu"):
            self.cache, self.allocator, _ = build_fixture(
                cfg, tree_core_backend="python"
            )

    def _insert(self, tokens, swa_evicted=0):
        self.cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=self.allocator.alloc(len(tokens))[: len(tokens)],
                component_evicted_seqlens={SWA: swa_evicted} if swa_evicted else {},
            )
        )

    def _leaf(self, tokens):
        return self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        ).last_device_node

    def _swa_lock(self, end):
        """SWA lock_ref of the path node ending at token end."""
        node = self.cache.tree_core.node_by_id(self._leaf(list(range(1, end + 1))))
        return node.component_data[SWA].lock_ref


class TestDecodePullable(_Base):
    def test_reads_past_the_match_cap_up_to_the_last_prompt_token(self):
        self._insert(list(range(1, 17)), swa_evicted=8)
        # Prompt of 20: the pull may end at 19, the tree holds 16.
        pullable = decode_pullable(self.cache, _req(range(1, 21)))
        self.assertEqual(pullable.length, 16)
        # A prompt ending inside the tree stops one token short of its end.
        self.assertEqual(decode_pullable(self.cache, _req(range(1, 14))).length, 12)

    def test_a_missing_window_is_not_pullable(self):
        self._insert(list(range(1, 17)), swa_evicted=14)
        self.assertEqual(decode_pullable(self.cache, _req(range(1, 21))).length, 0)

    def test_divergent_prompt_stops_at_the_branch(self):
        self._insert(list(range(1, 17)))
        prompt = list(range(1, 11)) + [99] * 10
        self.assertEqual(decode_pullable(self.cache, _req(prompt)).length, 10)


class TestPullHolds(_Base):
    def test_hold_locks_the_window_until_released(self):
        self._insert(list(range(1, 17)))
        holds = PullHolds()
        req = _req(range(1, 21))
        self.assertTrue(
            holds.acquire(self.cache, req, decode_pullable(self.cache, req))
        )
        self.assertEqual(self._swa_lock(16), 1)
        holds.release(req.rid)
        holds.release(req.rid)
        self.assertEqual(self._swa_lock(16), 0)

    def test_hold_splits_at_a_pullable_end_inside_a_node(self):
        self._insert(list(range(1, 17)))
        holds = PullHolds()
        req = _req(range(1, 21))
        node_id = self._leaf(list(range(1, 17)))
        self.assertTrue(holds.acquire(self.cache, req, PullablePrefix(12, node_id)))
        self.assertEqual(self._swa_lock(12), 1)
        holds.release(req.rid)
        self.assertEqual(self._swa_lock(12), 0)

    def test_hold_records_the_pages_a_pull_reads_by_room(self):
        self._insert(list(range(1, 17)))
        holds = PullHolds()
        req = _req(range(1, 21), room=77)
        holds.acquire(self.cache, req, decode_pullable(self.cache, req))
        held = holds.prefix_for_room(77)
        # Path values root first; the walk above may have split the leaf.
        full, swa = [], []
        node = self.cache.tree_core.node_by_id(self._leaf(list(range(1, 17))))
        while node.parent is not None:
            full = node.component_data[ComponentType.FULL].value.tolist() + full
            swa = node.component_data[SWA].value.tolist() + swa
            node = node.parent
        self.assertEqual(held.end, 16)
        self.assertEqual(list(held.full_pages), full)
        # Window [12, 16) at page size 1.
        self.assertEqual(held.window_start, 12)
        self.assertEqual(list(held.swa_pages), swa[-4:])
        self.assertEqual(held.full_pages_from(10), full[10:])
        holds.release(req.rid)
        self.assertIsNone(holds.prefix_for_room(77))

    def test_nothing_pullable_holds_nothing(self):
        holds = PullHolds()
        self.assertFalse(holds.acquire(self.cache, _req([1, 2]), PullablePrefix(0)))
        self.assertNotIn("r", holds)


class TestPromptEndAnchors(_Base):
    def test_next_turn_supersedes_and_budget_evicts_lru(self):
        turn1 = list(range(1, 9))
        turn2 = turn1 + list(range(50, 58))
        other = list(range(100, 108))
        for seq in (turn2, other):
            self._insert(seq)
        anchors = PromptEndAnchors()
        n1 = prompt_end_anchor_node(
            self.cache, _req(turn1, last_node=self._leaf(turn1))
        )
        anchors.pin(self.cache, n1.id, budget=2)
        self.assertEqual(self._swa_lock(8), 1)
        n2 = prompt_end_anchor_node(
            self.cache, _req(turn2, last_node=self._leaf(turn2))
        )
        anchors.pin(self.cache, n2.id, budget=2)
        # The second turn's anchor released the first's.
        self.assertEqual(anchors.held(self.cache), [n2.id])
        self.assertEqual(self._swa_lock(8), 0)
        n3 = self.cache.tree_core.node_by_id(self._leaf(other))
        anchors.pin(self.cache, n3.id, budget=1)
        self.assertEqual(anchors.held(self.cache), [n3.id])
        self.assertTrue(anchors.release_lru(self.cache))
        self.assertFalse(anchors.release_lru(self.cache))
        self.assertEqual(n3.component_data[SWA].lock_ref, 0)

    def test_no_anchor_without_a_resident_window(self):
        prompt = list(range(1, 9))
        self._insert(prompt, swa_evicted=8)
        req = _req(prompt + [9], last_node=self._leaf(prompt))
        self.assertIsNone(prompt_end_anchor_node(self.cache, req))


class TestHooks(_Base):
    def test_match_hook_reports_and_holds_only_when_enabled(self):
        self._insert(list(range(1, 17)))
        queue = SimpleNamespace(tree_cache=self.cache)
        match = SimpleNamespace(decode_prefix_len=8)
        for enabled, rid in ((False, "a"), (True, "b")):
            req = _req(range(1, 21), rid=rid, room=40 + enabled)
            with envs.SGLANG_TPU_PD_PREFIX_PULL.override(enabled):
                pd_prefix_pull.on_decode_match(match, queue, req)
            report = pullable_by_room.report(req.bootstrap_room, 8)
            self.assertEqual((report.pullable_len, report.pull_held), (16, enabled))
            self.assertEqual(rid in pd_prefix_pull.pull_holds, enabled)
        pd_prefix_pull.release_hold_with_kv(None, _req([], rid="b"))
        self.assertEqual(self._swa_lock(16), 0)

    def test_warmup_request_is_neither_reported_nor_held(self):
        from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

        self._insert(list(range(1, 17)))
        queue = SimpleNamespace(tree_cache=self.cache)
        req = _req(range(1, 21), rid="w", room=50, host=FAKE_BOOTSTRAP_HOST)
        with envs.SGLANG_TPU_PD_PREFIX_PULL.override(True):
            pd_prefix_pull.on_decode_match(
                SimpleNamespace(decode_prefix_len=8), queue, req
            )
        self.assertNotIn("w", pd_prefix_pull.pull_holds)
        self.assertEqual(pullable_by_room.report(50, 0).pullable_len, 0)
        self.assertEqual(self._swa_lock(16), 0)

    def test_hooks_registered_and_targets_resolve(self):
        for target, hook, hook_type in pd_prefix_pull.hook_specs():
            self.assertIn((target, hook, hook_type), HOOKS)
            pkgutil.resolve_name(target)


if __name__ == "__main__":
    unittest.main()
