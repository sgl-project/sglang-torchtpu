"""Lower-priority unit tests of the victim push (sglang_tpu/pd_victim_push.py): the wire round trip, less common pusher and receiver paths, the decode's flush gate, and the write-through host reclaim and eviction guard. Fixtures come from test_pd_victim_push.py."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import pd_host_prefix_recall
from sglang_tpu import pd_victim_push as victim_push
from sglang_tpu.disaggregation.raiden.serde import VictimAck, VictimDone, VictimPush

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _load_sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_main = _load_sibling("test_pd_victim_push")
FULL = _main.FULL
_DecodeReceiverFixture = _main._DecodeReceiverFixture
_PrefillPusherFixture = _main._PrefillPusherFixture
device_hit = _main.device_hit
full_cache = _main.full_cache
host_hit = _main.host_hit
insert_tokens = _main.insert_tokens
root_node = _main.root_node


class TestWire(CustomTestCase):
    def test_push_ack_done_round_trip(self):
        for msg in (
            VictimPush(
                prefill="ab12",
                seq=3,
                ack_endpoint="tcp://10.0.0.9:7000",
                tokens=[1, 2],
                blocks=[4, 5],
                uuid=9,
                address="10.0.0.1:9000",
            ),
            VictimAck(seq=3, stored=True),
            VictimAck(seq=4, stored=False),
            VictimDone(prefill="ab12", seq=3, ok=True),
        ):
            self.assertEqual(type(msg).from_frames(msg.to_frames()), msg)


class TestPrefillPusherExtra(_PrefillPusherFixture, CustomTestCase):
    def test_declined_offer_retains_registration_pin_until_native_failed(self):
        sent = self.offer_once()
        self.acks[sent["seq"]] = VictimAck(seq=sent["seq"], stored=False)
        self.pusher.step()
        self.assertIn(sent["seq"], self.pusher._pushes)
        self.cache.evict_host(32, FULL)
        self.assertEqual(host_hit(self.cache, self.a), 8)
        self.host.transfer.complete("failed")
        self.pusher.step()
        self.assertNotIn(sent["seq"], self.pusher._pushes)
        self.assertTrue(
            all(
                not call.kwargs["ok"]
                for call in self.mgr.send_victim_done.call_args_list
            )
        )
        self.cache.evict_host(32, FULL)
        self.assertEqual(host_hit(self.cache, self.a), 0)

    def test_ack_timeout_does_not_release_pages_still_registered_with_native(self):
        sent = self.offer_once()
        with patch.object(victim_push, "_ACK_TIMEOUT_S", -1):
            self.pusher.step()
        self.assertIn(sent["seq"], self.pusher._pushes)
        self.cache.evict_host(32, FULL)
        self.assertEqual(host_hit(self.cache, self.a), 8)
        self.host.transfer.complete("failed")
        self.pusher.step()
        self.assertNotIn(sent["seq"], self.pusher._pushes)

    def test_cache_flush_refuses_to_release_a_registered_source(self):
        self.offer_once()
        with patch.object(victim_push, "_DRAIN_TIMEOUT_S", 0):
            with self.assertRaises(RuntimeError):
                self.pusher.release_all()
        self.cache.evict_host(32, FULL)
        self.assertEqual(host_hit(self.cache, self.a), 8)
        self.host.transfer.complete("failed")
        self.pusher.release_all()
        self.cache.evict_host(32, FULL)
        self.assertEqual(host_hit(self.cache, self.a), 0)

    def test_false_registration_result_is_polled_to_terminal(self):
        """Native false can mean an already-ACKed registration, not rejection."""
        self.host.transfer.registration_result = False
        sent = self.offer_once()
        self.acks[sent["seq"]] = VictimAck(seq=sent["seq"], stored=True)
        self.host.transfer.complete("sent")
        self.pusher.step()
        self.assertEqual(self.pusher.stats["stored"], 1)

    def test_device_only_session_is_not_offered(self):
        c = list(range(300, 308))
        insert_tokens(self.cache, self.allocator, c)
        self.peers[3] = ("10.0.0.3", 5003)
        self.pusher.record(SimpleNamespace(bootstrap_room=3, origin_input_ids=c))
        for _ in range(3):
            self.pusher.step()
            self.host.transfer.complete("failed")
            seq = self.mgr.send_victim_push.call_args.kwargs["seq"]
            self.acks[seq] = VictimAck(seq=seq, stored=False)
        offered = [c.kwargs["tokens"] for c in self.mgr.send_victim_push.call_args_list]
        self.assertNotIn(c, offered)

    def test_no_offer_without_host_pressure(self):
        with patch.object(victim_push, "_PUSH_FULL_FREE_TOKENS", 0):
            for _ in range(3):
                self.pusher.step()
        self.mgr.send_victim_push.assert_not_called()
        self.assertFalse(self.host.transfer.registrations)


class TestDecodeReceiverExtra(_DecodeReceiverFixture, CustomTestCase):
    def test_native_receive_waits_for_producer_done_before_publication(self):
        self.granted()
        self.host.transfer.complete("received")
        self.round()
        self.assertEqual(host_hit(self.cache, self.c), 0)
        self.assertTrue(self.receiver.holds_pages())
        self.done()
        self.round()
        self.assertEqual(host_hit(self.cache, self.c), 8)

    def test_landing_timeout_does_not_reuse_pages_a_remote_writer_may_hold(self):
        self.granted()
        with patch.object(victim_push, "_LANDING_TIMEOUT_S", -1):
            self.round()
        self.assertTrue(self.receiver.holds_pages())
        self.assertEqual(self.host.used, 8)
        self.host.transfer.complete("failed")
        self.round()
        self.assertFalse(self.receiver.holds_pages())
        self.assertEqual(self.host.used, 0)

    def test_cache_flush_refuses_to_reuse_a_pending_host_landing(self):
        self.granted()
        with patch.object(victim_push, "_DRAIN_TIMEOUT_S", 0):
            with self.assertRaises(RuntimeError):
                self.receiver.drain()
        self.assertEqual(self.host.used, 8)
        self.host.transfer.complete("failed")
        self.receiver.drain()
        self.assertEqual(self.host.used, 0)


class TestDecodeFlushWaitsForDeferredReleases(CustomTestCase):
    """A decode cache reset must wait for pages the transfer queue still owns."""

    def _scheduler(self, *, deferred: bool, mode=DisaggregationMode.DECODE):
        queue = SimpleNamespace(has_pending_deferred_releases=lambda: deferred)
        return SimpleNamespace(
            disaggregation_mode=mode, disagg_decode_transfer_queue=queue
        )

    def test_pinned_offers_keep_the_decode_non_idle_for_a_flush(self):
        upstream = MagicMock(return_value=True)
        scheduler = self._scheduler(deferred=True)
        self.assertFalse(victim_push._is_fully_idle(upstream, scheduler))
        # Health checks only ask about running work, as upstream does.
        self.assertTrue(
            victim_push._is_fully_idle(upstream, scheduler, for_health_check=True)
        )
        upstream.assert_called_with(
            scheduler, for_health_check=True, ignore_waiting=False
        )

    def test_released_pages_and_other_roles_defer_to_upstream(self):
        upstream = MagicMock(return_value=True)
        self.assertTrue(
            victim_push._is_fully_idle(upstream, self._scheduler(deferred=False))
        )
        self.assertTrue(
            victim_push._is_fully_idle(
                upstream,
                self._scheduler(deferred=True, mode=DisaggregationMode.PREFILL),
            )
        )
        upstream.return_value = False
        self.assertFalse(
            victim_push._is_fully_idle(upstream, self._scheduler(deferred=False))
        )

    def test_scheduler_idle_check_is_wrapped(self):
        self.assertIn(
            (Scheduler, "is_fully_idle", victim_push._is_fully_idle),
            victim_push._patch_targets(),
        )


class TestRecallLandingReclaimsHostDuplicates(CustomTestCase):
    """A recall lands in the prefill's RAM; under write-through that RAM is
    full of copies of pages the device still holds, which host eviction
    never takes. The puller reclaims those copies for the landing."""

    def _write_through_cache(self):
        cache, allocator = full_cache(kv_size=64, host_capacity=16)
        cache.tree_core.is_write_back = cache.is_write_back = False
        tokens = list(range(100, 116))
        insert_tokens(cache, allocator, tokens)
        node = root_node(cache, 100)
        host = cache.cache_controller.host
        cache.tree_core.commit_backup(node.id, host.alloc(len(node.key)), {})
        # The write-through ack registers the duplicate in production.
        cache.tree_core._update_duplicate_tracking(node)
        return cache, host, tokens

    def test_host_only_eviction_frees_nothing_but_the_reclaim_does(self):
        cache, host, tokens = self._write_through_cache()
        self.assertEqual(host.available_size(), 0)
        self.assertEqual(cache.evict_host(8, FULL), 0)
        freed = pd_host_prefix_recall.reclaim_full_host_duplicates(cache, 8)
        self.assertGreaterEqual(freed, 8)
        self.assertGreaterEqual(host.available_size(), 8)
        # The device copy is untouched: the prompt still hits in HBM.
        self.assertEqual(device_hit(cache, tokens), 16)

    def test_a_duplicate_whose_children_are_on_device_is_taken(self):
        cache, host, tokens = self._write_through_cache()
        _, allocator = cache, cache.token_to_kv_pool_allocator
        insert_tokens(cache, allocator, tokens + [8] * 8)  # a device child
        self.assertGreaterEqual(
            pd_host_prefix_recall.reclaim_full_host_duplicates(cache, 8), 8
        )

    def test_a_duplicate_below_another_duplicate_is_taken_tip_first(self):
        # Write-through host prefixes stay closed under ancestors: the child's
        # copy goes first, the parent's only once no child has a host copy.
        cache, host, tokens = self._write_through_cache()
        _, allocator = cache, cache.token_to_kv_pool_allocator
        host.capacity += 8
        child_tokens = tokens + [8] * 8
        insert_tokens(cache, allocator, child_tokens)
        child = root_node(cache, 100).children[
            next(iter(root_node(cache, 100).children))
        ]
        cache.tree_core.commit_backup(child.id, host.alloc(len(child.key)), {})
        cache.tree_core._update_duplicate_tracking(child)
        self.assertEqual(host.available_size(), 0)
        freed = pd_host_prefix_recall.reclaim_full_host_duplicates(cache, 8)
        self.assertEqual(freed, 8)  # the child's copy, not the parent's
        parent = root_node(cache, 100)
        self.assertTrue(parent.backuped)
        self.assertFalse(child.backuped)
        self.assertEqual(
            pd_host_prefix_recall.reclaim_full_host_duplicates(cache, 16), 16
        )  # now the parent is a host tip and goes too

    def test_combined_demand_takes_the_child_then_the_parent_in_one_call(self):
        # A landing larger than the first eligible tip: the pass that takes the
        # child makes the parent eligible, and the same call must take it too
        # instead of leaving the recall to fail with enough reclaimable RAM.
        cache, host, tokens = self._write_through_cache()
        _, allocator = cache, cache.token_to_kv_pool_allocator
        host.capacity += 8
        child_tokens = tokens + [8] * 8
        insert_tokens(cache, allocator, child_tokens)
        child = root_node(cache, 100).children[
            next(iter(root_node(cache, 100).children))
        ]
        cache.tree_core.commit_backup(child.id, host.alloc(len(child.key)), {})
        cache.tree_core._update_duplicate_tracking(child)
        self.assertEqual(host.available_size(), 0)
        self.assertEqual(
            pd_host_prefix_recall.reclaim_full_host_duplicates(cache, 16), 24
        )
        self.assertGreaterEqual(host.available_size(), 16)
        self.assertFalse(root_node(cache, 100).backuped)
        self.assertFalse(child.backuped)

    def test_a_duplicate_with_a_host_only_child_is_left_alone(self):
        # Deleting its device leaf later would orphan the subtree below it.
        cache, host, tokens = self._write_through_cache()
        from array import array as _array

        from sglang.srt.mem_cache.radix_cache import RadixKey as _RadixKey
        from sglang_tpu.pd_host_cache import insert_host_prefix

        host.capacity += 8
        insert_host_prefix(
            cache, _RadixKey(_array("q", tokens + [7] * 8)), host.alloc(8), 1
        )
        self.assertEqual(
            pd_host_prefix_recall.reclaim_full_host_duplicates(cache, 8), 0
        )

    def test_reclaim_is_a_no_op_without_pressure_or_the_walk(self):
        cache, host, _ = self._write_through_cache()
        self.assertEqual(
            pd_host_prefix_recall.reclaim_full_host_duplicates(cache, 0), 0
        )
        self.assertEqual(host.available_size(), 0)
        bare = SimpleNamespace(tree_core=SimpleNamespace())
        self.assertEqual(pd_host_prefix_recall.reclaim_full_host_duplicates(bare, 8), 0)


class TestEvictionWalkSkipsDeletedLeaves(CustomTestCase):
    """A leaf the eviction walk deleted earlier in the same walk may be
    yielded again; the wrapper skips it instead of raising KeyError."""

    def test_missing_leaf_is_skipped_and_present_leaf_is_evicted(self):
        cache, allocator = full_cache(kv_size=32, host_capacity=32)
        insert_tokens(cache, allocator, list(range(100, 108)))
        node = root_node(cache, 100)
        upstream = MagicMock(return_value="evicted")
        self.assertEqual(
            victim_push._evict_device_leaf(upstream, cache, node.id, {}), "evicted"
        )
        upstream.assert_called_once_with(cache, node.id, {})
        upstream.reset_mock()
        self.assertIsNone(victim_push._evict_device_leaf(upstream, cache, 10**9, {}))
        upstream.assert_not_called()

    def _unbacked_parent_of_a_host_only_child(self):
        cache, allocator = full_cache(kv_size=32, host_capacity=32)
        tokens = list(range(100, 108))
        insert_tokens(cache, allocator, tokens)
        node = root_node(cache, 100)
        host = cache.cache_controller.host
        from array import array as _array

        from sglang.srt.mem_cache.radix_cache import RadixKey as _RadixKey
        from sglang_tpu.pd_host_cache import insert_host_prefix

        insert_host_prefix(
            cache, _RadixKey(_array("q", tokens + [7] * 8)), host.alloc(8), 1
        )
        self.assertFalse(node.backuped)
        return cache, node

    def test_unbacked_parent_of_a_host_only_child_stays_on_device_write_through(self):
        cache, node = self._unbacked_parent_of_a_host_only_child()
        cache.tree_core.is_write_back = cache.is_write_back = False
        upstream = MagicMock(return_value="evicted")
        self.assertIsNone(victim_push._evict_device_leaf(upstream, cache, node.id, {}))
        upstream.assert_not_called()

    def test_unbacked_parent_of_a_host_only_child_is_backed_up_under_write_back(self):
        # write_back backs the node up before the demote (no orphan), so the
        # wrapper must let the step run: skipping it leaves every session's
        # leaf on the device after a recall landing and the walk frees nothing.
        cache, node = self._unbacked_parent_of_a_host_only_child()
        self.assertTrue(cache.is_write_back)
        upstream = MagicMock(return_value="backup")
        self.assertEqual(
            victim_push._evict_device_leaf(upstream, cache, node.id, {}), "backup"
        )
        upstream.assert_called_once_with(cache, node.id, {})

    def test_walk_is_wrapped(self):
        self.assertIn(
            (UnifiedRadixCache, "_evict_device_leaf", victim_push._evict_device_leaf),
            victim_push._patch_targets(),
        )


if __name__ == "__main__":
    unittest.main()
