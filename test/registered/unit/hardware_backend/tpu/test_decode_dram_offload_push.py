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
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.decode_dram_offload import (
    dram_offload_hooks,
    prefill_dram_victim_pusher,
    radix_host_ops,
)
from sglang_tpu.disaggregation.decode_dram_offload.dram_offload_extension import (
    DecodeDramOffloadExtension,
    PrefillDramOffloadExtension,
)
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
        self.assertEqual(registrations, [(sent["uuid"], sent["blocks"], "host")])
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
        self.pusher.note_decode_of(pushed_req(room=9, tokens=forked))
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
        pusher.note_decode_of(pushed_req(room=1, tokens=session))
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
        pusher.note_decode_of(pushed_req(room=1, tokens=session))
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
        return queue, patch.object(dram_offload_hooks, "_extension", return_value=extension)

    def test_queue_step_is_driven_before_and_settled_after(self):
        for returned, bootstrapped in ((["a"], ["a"]), ((["a"], ["failed"]), ["a"])):
            with self.subTest(returned=returned):
                calls = []
                extension = MagicMock()
                extension.before_bootstrap_step.side_effect = lambda *a, **k: (
                    calls.append("advance")
                )
                extension.after_bootstrap_step.side_effect = lambda **k: calls.append(
                    "after"
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
                self.assertEqual(calls, ["advance", "pop", "after"])
                extension.before_bootstrap_step.assert_called_once_with(
                    queue.queue, waiting_reqs=["waiting"], serving=True
                )
                extension.after_bootstrap_step.assert_called_once_with(
                    bootstrapped=bootstrapped, queued=queue.queue
                )
                upstream.assert_called_once_with(queue, return_failed_reqs=True)

    def test_engine_serves_while_a_request_is_queued_or_running(self):
        for running in (False, True):
            with self.subTest(running=running):
                extension = MagicMock()
                queue, found = self._queue(extension, queued=False, running=running)
                with found:
                    dram_offload_hooks._pop_bootstrapped(MagicMock(return_value=[]), queue)
                extension.before_bootstrap_step.assert_called_once_with(
                    [], waiting_reqs=[], serving=running
                )

    def test_memory_release_first_lets_go_of_the_pages_of_the_prefill_end(self):
        calls = []
        extension = MagicMock()
        extension.release_all.side_effect = lambda: calls.append("release")
        upstream = MagicMock(side_effect=lambda q: calls.append("upstream"))
        queue, found = self._queue(extension)
        with found:
            dram_offload_hooks._release_memory_occupation(upstream, queue)
        self.assertEqual(calls, ["release", "upstream"])

    def test_paused_engine_still_drains_and_is_not_idle_while_pages_are_held(self):
        extension = MagicMock(spec=PrefillDramOffloadExtension)
        scheduler = SimpleNamespace(
            tree_cache="tree", disaggregation_mode=DisaggregationMode.PREFILL
        )
        with patch.object(dram_offload_hooks, "_extension", return_value=extension):
            dram_offload_hooks._record_scheduler_state_for_paused_engine(MagicMock(), scheduler)
            extension.settle_host_transfers.assert_called_once_with()
            upstream = MagicMock(return_value=True)
            extension.holds_host_pages_outside_tree.return_value = True
            self.assertFalse(dram_offload_hooks._is_fully_idle(upstream, scheduler))
            # A health check asks about running requests only, or a probe of
            # an engine that only pushes would wait for a request to answer.
            self.assertTrue(
                dram_offload_hooks._is_fully_idle(upstream, scheduler, for_health_check=True)
            )
            extension.holds_host_pages_outside_tree.return_value = False
            self.assertTrue(dram_offload_hooks._is_fully_idle(upstream, scheduler))
            # Busy for core's own reasons: the extension is not asked.
            upstream.return_value = False
            self.assertFalse(dram_offload_hooks._is_fully_idle(upstream, scheduler))
        # An engine without the prefill end is left to core.
        with patch.object(dram_offload_hooks, "_extension", return_value=None):
            upstream = MagicMock()
            dram_offload_hooks._record_scheduler_state_for_paused_engine(upstream, scheduler)
            upstream.assert_called_once_with(scheduler)

    def test_paused_decode_settles_its_landings_so_that_it_can_be_flushed(self):
        # The decode queues are not stepped while paused; without this tick a
        # granted landing keeps the engine non-idle for the whole pause.
        extension = MagicMock(spec=DecodeDramOffloadExtension)
        scheduler = SimpleNamespace(
            tree_cache="tree", disaggregation_mode=DisaggregationMode.DECODE
        )
        with patch.object(dram_offload_hooks, "_extension", return_value=extension):
            dram_offload_hooks._record_scheduler_state_for_paused_engine(MagicMock(), scheduler)
        extension.settle_host_transfers.assert_called_once_with()


class TestOffloadIsAnExtensionOfTheL2Controller(CustomTestCase):
    """The decode DRAM offload hangs off the cache's L2 controller, so the
    cache dram_offload_hooks reach every worker through it and a cache without it is left
    alone."""

    def test_cache_reset_first_lets_go_of_every_page_the_extension_holds(self):
        cache, _ = full_cache(host_capacity=32)
        calls = []
        offload = MagicMock()
        offload.before_cache_reset.side_effect = lambda: calls.append("release")
        upstream = MagicMock(side_effect=lambda c: calls.append("reset"))
        # Without the extension the reset is upstream's.
        dram_offload_hooks._reset_cache(upstream, cache)
        self.assertEqual(calls, ["reset"])
        cache.cache_controller.decode_dram_extension = offload
        dram_offload_hooks._reset_cache(upstream, cache)
        self.assertEqual(calls, ["reset", "release", "reset"])
        # A refusal (a peer may still read or write a page) stops the reset.
        offload.before_cache_reset.side_effect = RuntimeError("pinned")
        with self.assertRaises(RuntimeError):
            dram_offload_hooks._reset_cache(upstream, cache)
        self.assertEqual(upstream.call_count, 2)

    def test_reset_lets_go_of_the_pages_of_either_role(self):
        prefill = PrefillDramOffloadExtension.__new__(PrefillDramOffloadExtension)
        prefill.push = MagicMock()
        prefill.before_cache_reset()
        prefill.push.release_all.assert_called_once_with()
        prefill.push.clear_push_routes.assert_called_once_with()
        decode = DecodeDramOffloadExtension.__new__(DecodeDramOffloadExtension)
        decode.landing = MagicMock()
        decode.before_cache_reset()
        decode.landing.drain_host_landings.assert_called_once_with()

    def test_checkpoint_tells_the_pusher_which_decode_served_the_branch(self):
        cache, _ = full_cache(host_capacity=32)
        prefill = PrefillDramOffloadExtension.__new__(PrefillDramOffloadExtension)
        prefill.push = MagicMock()
        cache.cache_controller.decode_dram_extension = prefill
        upstream = MagicMock(return_value="done")
        self.assertEqual(dram_offload_hooks._checkpoint(upstream, cache, "req"), "done")
        prefill.push.track_host_leaf.assert_called_once_with("req")


if __name__ == "__main__":
    unittest.main()
