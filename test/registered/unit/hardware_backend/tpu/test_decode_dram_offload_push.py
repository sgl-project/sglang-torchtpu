"""Unit tests for victim push (decode_dram_offload/prefill_dram_victim_pusher.py and decode_dram_victim_receiver.py).

The prefill offers the host-backed session with the oldest path end to the
decode that served it, registers its pinned host pages for a native host pull,
and makes its own copy the next eviction victim once the transfer landed.
The decode inserts the session as a host-only node. Real
full-attention unified trees with a fake raiden controller and host transfer;
the raiden manager is a fake.
"""

import heapq
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.decode_dram_offload import (
    dram_offload_hooks,
    prefill_dram_victim_pusher,
    radix_host_ops,
)
from sglang_tpu.disaggregation.decode_dram_offload.dram_offload_hooks import (
    DecodeDramOffloadExtension,
    PrefillDramOffloadExtension,
)
from sglang_tpu.disaggregation.raiden.serde import VictimAck, VictimPush
from sglang_tpu.testing.decode_dram_offload import (
    FULL,
    DecodeReceiverFixture,
    PrefillPusherFixture,
    device_hit,
    full_cache,
    host_hit,
    insert_tokens,
    offload_context,
    push_channel,
    pushed_req,
    root_node,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPrefillPusher(PrefillPusherFixture, CustomTestCase):
    def test_stored_victim_path_evicts_in_the_same_order_on_every_rank(self):
        """A stored victim's path must take unique access stamps, each older
        than every live node. Stamping the whole path with one value tied the
        eviction heap, whose ties pop in set-iteration (per-rank memory) order,
        so TP ranks evicted different nodes after the first stored push."""
        c = list(range(300, 308))
        insert_tokens(self.cache, self.allocator, c[:4])  # a two-node path
        insert_tokens(self.cache, self.allocator, c)
        leaf = root_node(self.cache, c[0])
        while leaf.children:
            leaf = next(iter(leaf.children.values()))
        radix_host_ops.mark_next_victim(self.cache, leaf.id, -1.0e15)
        tree_core = self.cache.tree_core
        victims, node = [], leaf
        while node is not tree_core.root_node:
            victims.append(node)
            node = node.parent
        self.assertGreaterEqual(len(victims), 2)
        live, stack = [], list(tree_core.root_node.children.values())
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if node not in victims:
                live.append(node)
        self.assertTrue(live)
        stamps = [node.last_access_time for node in victims]
        self.assertEqual(len(set(stamps)), len(stamps), "victim nodes tie")
        self.assertLess(max(stamps), min(node.last_access_time for node in live))
        # The eviction heap pops the same nodes whichever order a rank's leaf
        # set yields them in.
        entries = [(node.last_access_time, node) for node in victims + live]
        orders = []
        for candidates in (entries, entries[::-1]):
            heap = list(candidates)
            heapq.heapify(heap)
            orders.append([heapq.heappop(heap)[1].id for _ in range(len(heap))])
        self.assertEqual(orders[0], orders[1])

    def test_host_source_stays_pinned_until_native_sent_then_is_next_victim(self):
        sent = self.offer_once()
        self.assertEqual(sent["decode"], ("10.0.0.1", 5001))
        self.assertEqual(sent["tokens"], self.a)
        registrations = list(self.host.transfer.registrations.values())
        self.assertEqual(registrations, [(sent["uuid"], sent["blocks"])])
        # The decode stored the push; the native registration has not reported.
        self.replies[sent["seq"]] = True
        self.pusher.advance(serving=True)
        self.cache.evict_host(8, FULL)
        self.assertEqual(host_hit(self.cache, self.a), 8)
        self.assertEqual(self.pusher.stats["stored"], 0)
        self.host.transfer.complete("sent")
        self.pusher.advance(serving=True)
        self.assertEqual(self.pusher.stats["stored"], 1)
        self.mgr.forget_victim.assert_called_once_with(sent["seq"])
        self.cache.evict_host(8, FULL)
        self.assertEqual(host_hit(self.cache, self.a), 0)

    def test_no_push_starts_unless_the_engine_serves_requests(self):
        """An engine left alone settles the pushes in flight and starts no new
        one: it goes idle, so a cache flush is not refused for as long as
        there are branches to push."""
        self.pusher.advance(serving=False)
        self.mgr.send_victim_push.assert_not_called()
        sent = self.offer_once()
        self.replies[sent["seq"]] = True
        self.host.transfer.complete("sent")
        self.pusher.advance(serving=False)
        self.assertEqual(self.pusher.stats["stored"], 1)
        self.assertFalse(self.pusher.has_pushes_in_flight())
        self.mgr.send_victim_push.assert_called_once()

    def test_request_without_a_peer_is_matched_like_on_the_other_ranks(self):
        """The listener thread can drop a room's peer on one rank only. A
        match splits nodes, so that rank matches too and only routes nothing."""
        forked = self.a[:4] + [900, 901, 902, 903]
        self.assertEqual(len(root_node(self.cache, self.a[0]).key), 8)
        self.pusher.note_decode_of(pushed_req(room=9, tokens=forked), recalled=False)
        self.assertEqual(len(root_node(self.cache, self.a[0]).key), 4)
        self.assertEqual(len(self.pusher._routes), 2)

    def test_shared_prefix_of_a_push_in_flight_keeps_its_host_pages(self):
        """Under write-back any host eviction reclaims the host copy of a node
        that is also on the device and not host-locked. The decode reads pages
        of the whole pushed path, so every node of it stays locked."""
        cache, allocator = full_cache(kv_size=64, host_capacity=32)
        channel = push_channel(cache.cache_controller.host)
        channel.victim_reply.return_value = None
        channel.decode_peer.return_value = ("10.0.0.1", 5001)
        pusher = prefill_dram_victim_pusher.PrefillDramVictimPusher(
            offload_context(cache=cache, channel=channel)
        )
        session = [1, 2, 3, 4, 10, 11, 12, 13]
        insert_tokens(cache, allocator, session)
        cache.evict(EvictParams(num_tokens=8))
        # Another session brings the shared prefix back to the device.
        insert_tokens(cache, allocator, [1, 2, 3, 4, 20, 21, 22, 23])
        shared = root_node(cache, 1)
        self.assertIsNotNone(shared.component_data[FULL].value)
        pusher.note_decode_of(pushed_req(room=1, tokens=session), recalled=False)
        pusher.advance(serving=True)
        pushed = channel.send_victim_push.call_args.kwargs["blocks"]
        self.assertEqual(len(pushed), 8)
        cache.evict_host(4)
        self.assertEqual(
            radix_host_ops.host_pages(
                cache, cache.cache_controller.host, pusher._pushes[1].leaf
            ),
            pushed,
        )

    def test_leaf_whose_backup_is_pending_is_pushed_once_it_completed(self):
        """Under write-through a new leaf has its host pages reserved before
        the copy is done. Picking it then must not rule it out for good."""
        cache, allocator = full_cache(kv_size=64, host_capacity=32)
        cache.tree_core.is_write_back = cache.is_write_back = False
        cache.cache_controller.write_policy = "write_through"
        cache.write_through_threshold = 1
        channel = push_channel(cache.cache_controller.host)
        channel.victim_reply.return_value = None
        channel.decode_peer.return_value = ("10.0.0.1", 5001)
        pusher = prefill_dram_victim_pusher.PrefillDramVictimPusher(
            offload_context(cache=cache, channel=channel)
        )
        session = [1, 2, 3, 4, 10, 11, 12, 13]
        insert_tokens(cache, allocator, session)
        self.assertIsNotNone(root_node(cache, 1).write_through_pending_id)
        pusher.note_decode_of(pushed_req(room=1, tokens=session), recalled=False)
        pusher.advance(serving=True)
        channel.send_victim_push.assert_not_called()
        cache.cache_controller.start_writing()
        cache.writing_check()
        pusher.advance(serving=True)
        channel.send_victim_push.assert_called_once()

    def test_native_failure_declines_the_push_and_unlocks_the_host_source(self):
        sent = self.offer_once()
        self.replies[sent["seq"]] = True
        self.host.transfer.complete("failed")
        self.pusher.advance(serving=True)
        self.assertEqual(self.pusher.stats["declined"], 1)
        self.assertEqual(host_hit(self.cache, self.a), 8)


class TestDecodeReceiver(DecodeReceiverFixture, CustomTestCase):
    def test_push_is_inserted_and_answered_once_its_read_has_landed(self):
        self.granted()
        self.round()
        self.assertTrue(self.receiver.has_host_landings())
        self.assertEqual(host_hit(self.cache, self.c), 0)
        self.host.transfer.complete("received")
        self.round()
        self.assertFalse(self.receiver.has_host_landings())
        self.assertEqual(device_hit(self.cache, self.c), 0)
        self.assertEqual(host_hit(self.cache, self.c), 8)
        self.assertEqual(self.receiver.stats["landed_tokens"], 8)
        self.assertTrue(self.answered())
        self.assertEqual(
            self.mgr.send_victim_ack.call_args.kwargs["endpoint"], "tcp://p"
        )

    def test_failed_read_frees_its_pages_and_is_answered_as_not_stored(self):
        self.granted()
        self.assertEqual(self.host.used, 8)
        self.host.transfer.complete("failed")
        self.round()
        self.assertFalse(self.receiver.has_host_landings())
        self.assertEqual(self.host.used, 0)
        self.assertEqual(host_hit(self.cache, self.c), 0)
        self.assertFalse(self.answered())

    def test_full_host_pool_refuses(self):
        self.host.capacity = 4
        self.round()
        self.assertFalse(self.answered())
        self.assertFalse(self.receiver.has_host_landings())
        self.assertFalse(self.host.transfer.reads)


def _prefill_end():
    return PrefillDramOffloadExtension(recall=MagicMock(), push=MagicMock())


def _decode_end():
    return DecodeDramOffloadExtension(landing=MagicMock(), source=MagicMock())


class TestPrefillQueueHooks(CustomTestCase):
    """The plugin's wrappers of the prefill bootstrap queue and scheduler:
    core is unchanged, so every step of the prefill's end is driven here."""

    def _queue(self, extension, *, queued=True, running=False):
        queued = [SimpleNamespace(bootstrap_room=1)] if queued else []
        queue = SimpleNamespace(
            queue=queued,
            scheduler=SimpleNamespace(
                tree_cache="tree",
                waiting_queue=["waiting"] if queued else [],
                running_batch=SimpleNamespace(is_empty=lambda: not running),
            ),
        )
        return queue, patch.object(
            dram_offload_hooks, "_extension", return_value=extension
        )

    def test_queue_step_is_driven_before_and_settled_after(self):
        for returned in (["a"], (["a"], ["failed"])):
            with self.subTest(returned=returned):
                calls = []
                extension = _prefill_end()
                extension.recall.advance.side_effect = lambda *a, **k: calls.append(
                    "recall"
                )
                extension.push.advance.side_effect = lambda **k: calls.append("push")
                extension.recall.insert_recalled_host_prefix.side_effect = lambda req: (
                    calls.append("insert") or True
                )
                extension.push.note_decode_of.side_effect = lambda *a, **k: (
                    calls.append("note")
                )
                extension.recall.drop_departed.side_effect = lambda q: calls.append(
                    "drop"
                )
                upstream = MagicMock(
                    side_effect=lambda *a, **k: calls.append("pop") or returned
                )
                queue, found = self._queue(extension)
                with found:
                    result = dram_offload_hooks._pop_bootstrapped(
                        upstream, queue, return_failed_reqs=True
                    )
                self.assertEqual(result, returned)
                self.assertEqual(
                    calls, ["recall", "push", "pop", "insert", "note", "drop"]
                )
                extension.recall.advance.assert_called_once_with(
                    queue.queue, waiting_reqs=["waiting"]
                )
                extension.push.advance.assert_called_once_with(serving=True)
                extension.recall.insert_recalled_host_prefix.assert_called_once_with(
                    "a"
                )
                extension.push.note_decode_of.assert_called_once_with(
                    "a", recalled=True
                )
                extension.recall.drop_departed.assert_called_once_with(queue.queue)
                upstream.assert_called_once_with(queue, return_failed_reqs=True)

    def test_engine_serves_while_a_request_is_queued_or_running(self):
        for running in (False, True):
            with self.subTest(running=running):
                extension = _prefill_end()
                queue, found = self._queue(extension, queued=False, running=running)
                with found:
                    dram_offload_hooks._pop_bootstrapped(
                        MagicMock(return_value=[]), queue
                    )
                extension.push.advance.assert_called_once_with(serving=running)

    def test_memory_release_first_lets_go_of_the_pages_of_the_prefill_end(self):
        calls = []
        extension = _prefill_end()
        extension.push.release_all.side_effect = lambda: calls.append("push")
        extension.recall.release_all.side_effect = lambda: calls.append("recall")
        upstream = MagicMock(side_effect=lambda q: calls.append("upstream"))
        queue, found = self._queue(extension)
        with found:
            dram_offload_hooks._release_memory_occupation(upstream, queue)
        self.assertEqual(calls, ["push", "recall", "upstream"])

    def test_paused_engine_still_drains_and_is_not_idle_while_pages_are_held(self):
        extension = _prefill_end()
        extension.push.has_pushes_in_flight.return_value = False
        scheduler = SimpleNamespace(
            tree_cache="tree", disaggregation_mode=DisaggregationMode.PREFILL
        )
        with patch.object(dram_offload_hooks, "_extension", return_value=extension):
            dram_offload_hooks._record_scheduler_state_for_paused_engine(
                MagicMock(), scheduler
            )
            extension.recall.free_settled_host_landings.assert_called_once_with()
            extension.push.retire_settled_pushes.assert_called_once_with()
            upstream = MagicMock(return_value=True)
            extension.recall.holds_host_landing_pages.return_value = True
            self.assertFalse(dram_offload_hooks._is_fully_idle(upstream, scheduler))
            # A health check asks about running requests only, or a probe of
            # an engine that only pushes would wait for a request to answer.
            self.assertTrue(
                dram_offload_hooks._is_fully_idle(
                    upstream, scheduler, for_health_check=True
                )
            )
            extension.recall.holds_host_landing_pages.return_value = False
            self.assertTrue(dram_offload_hooks._is_fully_idle(upstream, scheduler))
            # Busy for core's own reasons: the extension is not asked.
            upstream.return_value = False
            self.assertFalse(dram_offload_hooks._is_fully_idle(upstream, scheduler))
        # An engine without the prefill end is left to core.
        with patch.object(dram_offload_hooks, "_extension", return_value=None):
            upstream = MagicMock()
            dram_offload_hooks._record_scheduler_state_for_paused_engine(
                upstream, scheduler
            )
            upstream.assert_called_once_with(scheduler)

    def test_memory_release_for_a_weight_update_waits_for_transfers_in_flight(self):
        """The weight updater asserts an idle engine; pages held outside the
        tree right after the last request must not fail that assertion."""
        extension = _prefill_end()
        extension.recall.holds_host_landing_pages.side_effect = [True, True, False]
        extension.push.has_pushes_in_flight.return_value = False
        scheduler = SimpleNamespace(
            tree_cache="tree", disaggregation_mode=DisaggregationMode.PREFILL
        )
        updater = SimpleNamespace(scheduler=scheduler)
        upstream = MagicMock(return_value="released")
        with (
            patch.object(dram_offload_hooks, "_extension", return_value=extension),
            patch.object(dram_offload_hooks.time, "sleep"),
        ):
            result = dram_offload_hooks._release_memory_occupation_weights(
                upstream, updater, "req"
            )
        self.assertEqual(result, "released")
        self.assertEqual(extension.push.retire_settled_pushes.call_count, 2)
        upstream.assert_called_once_with(updater, "req")

    def test_paused_decode_settles_its_landings_so_that_it_can_be_flushed(self):
        # The decode queues are not stepped while paused; without this tick a
        # granted landing keeps the engine non-idle for the whole pause.
        extension = _decode_end()
        scheduler = SimpleNamespace(
            tree_cache="tree", disaggregation_mode=DisaggregationMode.DECODE
        )
        with patch.object(dram_offload_hooks, "_extension", return_value=extension):
            dram_offload_hooks._record_scheduler_state_for_paused_engine(
                MagicMock(), scheduler
            )
        extension.source.settle_offers.assert_called_once_with()
        extension.landing.finish_agreed_landings.assert_called_once_with()


class TestOffloadIsAnExtensionOfTheL2Controller(CustomTestCase):
    """The decode DRAM offload hangs off the cache's L2 controller, so the
    cache hooks reach every worker through it and a cache without it is left
    alone."""

    def test_cache_reset_first_lets_go_of_every_page_the_extension_holds(self):
        cache, _ = full_cache(host_capacity=32)
        calls = []
        upstream = MagicMock(side_effect=lambda c: calls.append("reset"))
        # Without the extension the reset is upstream's.
        dram_offload_hooks._reset_cache(upstream, cache)
        self.assertEqual(calls, ["reset"])
        prefill, decode = _prefill_end(), _decode_end()
        for extension, workers in (
            (prefill, (prefill.push, prefill.recall)),
            (decode, (decode.landing, decode.source)),
        ):
            calls.clear()
            for order, worker in enumerate(workers):
                worker.release_all.side_effect = lambda order=order: calls.append(order)
            cache.cache_controller.decode_dram_extension = extension
            dram_offload_hooks._reset_cache(upstream, cache)
            self.assertEqual(calls, [0, 1, "reset"])
        # A refusal (a peer may still read or write a page) stops the reset.
        decode.landing.release_all.side_effect = RuntimeError("pinned")
        with self.assertRaises(RuntimeError):
            dram_offload_hooks._reset_cache(upstream, cache)
        self.assertEqual(upstream.call_count, 3)

    def test_checkpoint_tells_the_pusher_which_decode_served_the_branch(self):
        cache, _ = full_cache(host_capacity=32)
        prefill = _prefill_end()
        cache.cache_controller.decode_dram_extension = prefill
        upstream = MagicMock(return_value="done")
        self.assertEqual(dram_offload_hooks._checkpoint(upstream, cache, "req"), "done")
        prefill.push.track_host_leaf.assert_called_once_with("req")


# ---- Lower-priority cases: less common paths, wire round trips, wording ----


class TestWire(CustomTestCase):
    def test_push_and_ack_round_trip(self):
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
        ):
            self.assertEqual(type(msg).from_frames(msg.to_frames()), msg)


class TestPrefillPusherExtra(PrefillPusherFixture, CustomTestCase):
    def test_declined_offer_retains_registration_pin_until_native_failed(self):
        sent = self.offer_once()
        self.replies[sent["seq"]] = False
        self.pusher.advance(serving=True)
        self.assertIn(sent["seq"], self.pusher._pushes)
        self.cache.evict_host(32, FULL)
        self.assertEqual(host_hit(self.cache, self.a), 8)
        self.host.transfer.complete("failed")
        self.pusher.advance(serving=True)
        self.assertNotIn(sent["seq"], self.pusher._pushes)
        self.assertEqual(self.pusher.stats["stored"], 0)
        self.cache.evict_host(32, FULL)
        self.assertEqual(host_hit(self.cache, self.a), 0)

    def test_ack_timeout_does_not_release_pages_still_registered_with_native(self):
        sent = self.offer_once()
        with patch.object(prefill_dram_victim_pusher, "_ACK_TIMEOUT_S", -1):
            self.pusher.advance(serving=True)
        self.assertIn(sent["seq"], self.pusher._pushes)
        self.cache.evict_host(32, FULL)
        self.assertEqual(host_hit(self.cache, self.a), 8)
        self.host.transfer.complete("failed")
        self.pusher.advance(serving=True)
        self.assertNotIn(sent["seq"], self.pusher._pushes)

    def test_memory_release_refuses_while_a_push_is_in_flight(self):
        sent = self.offer_once()
        with self.assertRaisesRegex(RuntimeError, "not idle"):
            self.pusher.release_all()
        self.cache.evict_host(32, FULL)
        self.assertEqual(host_hit(self.cache, self.a), 8)
        self.replies[sent["seq"]] = False
        self.host.transfer.complete("failed")
        self.pusher.advance(serving=False)
        self.pusher.release_all()
        self.assertEqual(len(self.pusher._routes), 0)

    def test_device_only_session_is_not_offered(self):
        c = list(range(300, 308))
        insert_tokens(self.cache, self.allocator, c)
        self.peers[3] = ("10.0.0.3", 5003)
        self.pusher.note_decode_of(pushed_req(room=3, tokens=c), recalled=False)
        for _ in range(3):
            self.pusher.advance(serving=True)
            self.host.transfer.complete("failed")
            seq = self.mgr.send_victim_push.call_args.kwargs["seq"]
            self.replies[seq] = False
        offered = [c.kwargs["tokens"] for c in self.mgr.send_victim_push.call_args_list]
        self.assertNotIn(c, offered)

    def test_no_offer_without_host_pressure(self):
        with patch.object(prefill_dram_victim_pusher, "_PUSH_FULL_FREE_TOKENS", 0):
            for _ in range(3):
                self.pusher.advance(serving=True)
        self.mgr.send_victim_push.assert_not_called()
        self.assertFalse(self.host.transfer.registrations)


class TestDecodeReceiverExtra(DecodeReceiverFixture, CustomTestCase):
    def test_read_that_cannot_start_is_answered_as_not_stored(self):
        self.host.transfer.read_error = RuntimeError("injected")
        self.round()
        self.assertEqual(self.receiver.stats["granted"], 1)
        self.assertFalse(self.receiver.has_host_landings())
        self.assertEqual(self.host.used, 0)
        self.assertFalse(self.answered())

    def test_cache_reset_refuses_while_a_landing_is_in_flight(self):
        self.granted()
        with self.assertRaisesRegex(RuntimeError, "not idle"):
            self.receiver.release_all()
        self.assertEqual(self.host.used, 8)
        self.host.transfer.complete("failed")
        self.round()
        self.receiver.release_all()
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
        self.assertFalse(dram_offload_hooks._is_fully_idle(upstream, scheduler))
        # Health checks only ask about running work, as upstream does.
        self.assertTrue(
            dram_offload_hooks._is_fully_idle(
                upstream, scheduler, for_health_check=True
            )
        )
        upstream.assert_called_with(
            scheduler, for_health_check=True, ignore_waiting=False
        )

    def test_released_pages_and_other_roles_defer_to_upstream(self):
        upstream = MagicMock(return_value=True)
        self.assertTrue(
            dram_offload_hooks._is_fully_idle(upstream, self._scheduler(deferred=False))
        )
        self.assertTrue(
            dram_offload_hooks._is_fully_idle(
                upstream,
                self._scheduler(deferred=True, mode=DisaggregationMode.NULL),
            )
        )
        upstream.return_value = False
        self.assertFalse(
            dram_offload_hooks._is_fully_idle(upstream, self._scheduler(deferred=False))
        )

    def test_scheduler_idle_check_is_wrapped(self):
        self.assertIn(
            (Scheduler, "is_fully_idle", dram_offload_hooks._is_fully_idle),
            dram_offload_hooks._patch_targets(),
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
        freed = radix_host_ops.reclaim_full_host_duplicates(cache, 8)
        self.assertGreaterEqual(freed, 8)
        self.assertGreaterEqual(host.available_size(), 8)
        # The device copy is untouched: the prompt still hits in HBM.
        self.assertEqual(device_hit(cache, tokens), 16)

    def test_a_duplicate_whose_children_are_on_device_is_taken(self):
        cache, host, tokens = self._write_through_cache()
        _, allocator = cache, cache.token_to_kv_pool_allocator
        insert_tokens(cache, allocator, tokens + [8] * 8)  # a device child
        self.assertGreaterEqual(
            radix_host_ops.reclaim_full_host_duplicates(cache, 8), 8
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
        freed = radix_host_ops.reclaim_full_host_duplicates(cache, 8)
        self.assertEqual(freed, 8)  # the child's copy, not the parent's
        parent = root_node(cache, 100)
        self.assertTrue(parent.backuped)
        self.assertFalse(child.backuped)
        self.assertEqual(
            radix_host_ops.reclaim_full_host_duplicates(cache, 16), 16
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
        self.assertEqual(radix_host_ops.reclaim_full_host_duplicates(cache, 16), 24)
        self.assertGreaterEqual(host.available_size(), 16)
        self.assertFalse(root_node(cache, 100).backuped)
        self.assertFalse(child.backuped)

    def test_a_duplicate_with_a_host_only_child_is_left_alone(self):
        # Deleting its device leaf later would orphan the subtree below it.
        cache, host, tokens = self._write_through_cache()
        from array import array as _array

        from sglang.srt.mem_cache.radix_cache import RadixKey as _RadixKey

        host.capacity += 8
        radix_host_ops.insert_host_prefix(
            cache, _RadixKey(_array("q", tokens + [7] * 8)), host.alloc(8), 1
        )
        self.assertEqual(radix_host_ops.reclaim_full_host_duplicates(cache, 8), 0)

    def test_host_insert_hashes_pages_only_when_the_tree_keeps_hashes(self):
        # Hashes feed L3 storage and the cache linker only; computing them for
        # every recall loaded the native hash extension on every pull, which
        # rebuilt it each time on a host whose ranks share its build directory.
        cache, host, tokens = self._write_through_cache()
        from array import array as _array

        from sglang.srt.mem_cache.radix_cache import RadixKey as _RadixKey

        host.capacity += 16
        with patch.object(radix_host_ops, "get_hash_str") as hash_str:
            radix_host_ops.insert_host_prefix(
                cache, _RadixKey(_array("q", tokens + [7] * 8)), host.alloc(8), 1
            )
            hash_str.assert_not_called()
            cache.tree_core.enable_storage = True
            hash_str.return_value = ["h"] * 8
            radix_host_ops.insert_host_prefix(
                cache, _RadixKey(_array("q", tokens + [9] * 8)), host.alloc(8), 1
            )
            hash_str.assert_called_once()

    def test_reclaim_is_a_no_op_without_pressure(self):
        cache, host, _ = self._write_through_cache()
        self.assertEqual(radix_host_ops.reclaim_full_host_duplicates(cache, 0), 0)
        self.assertEqual(host.available_size(), 0)


class TestEvictionWalkSkipsDeletedLeaves(CustomTestCase):
    """A leaf the eviction walk deleted earlier in the same walk may be
    yielded again; the wrapper skips it instead of raising KeyError."""

    def test_missing_leaf_is_skipped_and_present_leaf_is_evicted(self):
        cache, allocator = full_cache(kv_size=32, host_capacity=32)
        insert_tokens(cache, allocator, list(range(100, 108)))
        node = root_node(cache, 100)
        upstream = MagicMock(return_value="evicted")
        self.assertEqual(
            dram_offload_hooks._evict_device_leaf(upstream, cache, node.id, {}),
            "evicted",
        )
        upstream.assert_called_once_with(cache, node.id, {})
        upstream.reset_mock()
        self.assertIsNone(
            dram_offload_hooks._evict_device_leaf(upstream, cache, 10**9, {})
        )
        upstream.assert_not_called()

    def _unbacked_parent_of_a_host_only_child(self):
        cache, allocator = full_cache(kv_size=32, host_capacity=32)
        tokens = list(range(100, 108))
        insert_tokens(cache, allocator, tokens)
        node = root_node(cache, 100)
        host = cache.cache_controller.host
        from array import array as _array

        from sglang.srt.mem_cache.radix_cache import RadixKey as _RadixKey

        # Core's host insert alone: insert_host_prefix would back the parent
        # up with the received pages, which is the shape this walk must still
        # handle when it arises otherwise (a reclaimed duplicate, a failed
        # backup).
        key = _RadixKey(_array("q", tokens + [7] * 8))
        cache.tree_core.insert_host(
            cache.tree_core.root_node.id,
            key,
            host.alloc(16),
            radix_host_ops._page_hashes(cache.tree_core, list(key.token_ids), 1),
        )
        self.assertFalse(node.backuped)
        return cache, node

    def test_unbacked_parent_of_a_host_only_child_stays_on_device_write_through(self):
        cache, node = self._unbacked_parent_of_a_host_only_child()
        cache.tree_core.is_write_back = cache.is_write_back = False
        upstream = MagicMock(return_value="evicted")
        self.assertIsNone(
            dram_offload_hooks._evict_device_leaf(upstream, cache, node.id, {})
        )
        upstream.assert_not_called()

    def test_unbacked_parent_of_a_host_only_child_is_backed_up_under_write_back(self):
        # write_back backs the node up before the demote (no orphan), so the
        # wrapper must let the step run: skipping it leaves every session's
        # leaf on the device after a recall landing and the walk frees nothing.
        cache, node = self._unbacked_parent_of_a_host_only_child()
        self.assertTrue(cache.is_write_back)
        upstream = MagicMock(return_value="backup")
        self.assertEqual(
            dram_offload_hooks._evict_device_leaf(upstream, cache, node.id, {}),
            "backup",
        )
        upstream.assert_called_once_with(cache, node.id, {})

    def test_walk_is_wrapped(self):
        self.assertIn(
            (
                UnifiedRadixCache,
                "_evict_device_leaf",
                dram_offload_hooks._evict_device_leaf,
            ),
            dram_offload_hooks._patch_targets(),
        )


if __name__ == "__main__":
    unittest.main()
