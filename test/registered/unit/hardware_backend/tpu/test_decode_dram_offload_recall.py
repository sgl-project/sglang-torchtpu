"""Edge cases of the recall from decode RAM into the prefill's RAM
(sglang_tpu.prefill_dram_prefix_puller.PrefillDramPrefixPuller): no host room, room made
by reclaiming copies the device still holds, a READY that does not match the
plan, and a read that fails after it started. Every path must leave the
prefill computing the prompt, with no page leaked and no lock held."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec, patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.decode_dram_offload import (
    core_hooks,
    decode_dram_victim_receiver,
    radix_host_ops,
    startup_validation,
)
from sglang_tpu.disaggregation.decode_dram_offload.decode_dram_prefix_offerer import (
    DecodeDramPrefixOfferer,
)
from sglang_tpu.disaggregation.decode_dram_offload.dram_offload_extension import (
    DecodeDramOffloadExtension,
    PrefillDramOffloadExtension,
)
from sglang_tpu.disaggregation.decode_dram_offload.rank_lockstep import (
    DramOffloadWorkerContext,
    RankLockstep,
)
from sglang_tpu.disaggregation.raiden.dram_offload_channel import DramOffloadChannel
from sglang_tpu.testing.decode_dram_offload import (
    FULL,
    PREFILL_INFO,
    ROOM,
    SWITCH,
    DecodeQueueHooksFixture,
    HostRecallEdgeCasesFixture,
    RecallLifecycleFixture,
    decode_manager,
    full_cache,
    host_hit,
    insert_tokens,
    offload_context,
    prefill_manager,
    pushed_req,
    sender_on,
)

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestRecallLifecycle(
    RecallLifecycleFixture, HostRecallEdgeCasesFixture, CustomTestCase
):
    """The scheduler-step lifecycle of one recall on the prefill: offer, plan,
    allocate at READY, read, agree, insert, and the host pin that keeps the
    recalled pages until admission has taken the request."""

    def test_recall_runs_from_offer_to_a_pinned_host_node(self):
        mgr = self._scripted()
        cache, _, puller = self._build(host_capacity=32, mgr=mgr)
        host = cache.cache_controller.host
        req = self._req()
        plan = self._advance_to_pulling(puller, mgr, req)
        # The room stays in bootstrap while the read runs.
        puller.advance([req])
        self.assertIsNone(plan.agreed_done)
        mgr.resolve_prefix_pull.assert_not_called()
        mgr.read_done = True
        puller.advance([req])
        self.assertTrue(plan.agreed_done)
        mgr.resolve_prefix_pull.assert_called_once_with(ROOM)
        puller.insert_recalled_host_prefix(req)
        self.assertEqual(puller.stats["pulls"], 1)
        self.assertEqual(puller.stats["pulled_tokens"], 16)
        self.assertEqual(host.available_size(), 16)  # the landing joined the tree
        # The recalled pages cannot be evicted while the request waits ...
        self.assertIn(ROOM, puller._pins)
        self.assertEqual(cache.evict_host(16), 0)
        puller.advance([], waiting_reqs=[req])
        self.assertIn(ROOM, puller._pins)
        # ... and can once admission has taken it from the queues.
        puller.advance([])
        self.assertNotIn(ROOM, puller._pins)
        self.assertEqual(cache.evict_host(16), 16)

    def test_ranks_agree_on_the_range_before_the_decode_is_asked(self):
        """A rank that holds more of the prefix than another still pulls from
        the shortest local prefix, so every rank reads the same pages; the
        pages it already has are freed at the insert."""
        mgr = self._scripted()
        cache, _, puller = self._build(host_capacity=32, mgr=mgr)
        host = cache.cache_controller.host
        req = self._req()
        # This rank already holds the first 8 pages in host RAM.
        radix_host_ops.insert_host_prefix(
            cache, RadixKey(array("q", req.origin_input_ids[:8])), host.alloc(8), 1
        )

        def another_rank_holds_four_pages(rows, op):
            # The range agreement: one row of [proposed, want, -want, first
            # page] per room, then the budget row.
            return [[1, 16, -16, 4], rows[1]] if len(rows) == 2 else rows

        with patch.object(
            puller._lockstep, "reduce", side_effect=another_rank_holds_four_pages
        ):
            puller.advance([req])
        mgr.ask_prefix_pull.assert_called_once_with(ROOM, 4, 12)
        plan = puller._plans[ROOM]
        self.assertEqual((plan.first_page, plan.num_pages), (4, 12))
        self.assertEqual(plan.remote, list(range(4, 16)))
        mgr.ready, mgr.read_done = (11, "10.0.0.2:5"), None
        puller.advance([req])
        mgr.read_done = True
        puller.advance([req])
        puller.insert_recalled_host_prefix(req)
        self.assertEqual(puller.stats["pulls"], 1)
        # 8 pages held before plus the 8 new ones; the 4 read twice are freed.
        self.assertEqual(host.available_size(), 16)

    def test_no_rank_asks_unless_every_rank_wants_the_same_pages(self):
        for agreed in ([0, 0, 0, 0], [1, 8, -16, 0]):
            with self.subTest(agreed=agreed):
                mgr = self._scripted()
                cache, _, puller = self._build(host_capacity=32, mgr=mgr)
                with patch.object(
                    puller._lockstep,
                    "reduce",
                    side_effect=lambda rows, op: (
                        [agreed, rows[1]] if len(rows) == 2 else rows
                    ),
                ):
                    puller.advance([self._req()])
                mgr.ask_prefix_pull.assert_not_called()
                self.assertNotIn(ROOM, puller._plans)
                mgr.resolve_prefix_pull.assert_called_once_with(ROOM)
                self.assertFalse(puller.holds_host_landing_pages())

    def test_abort_between_ready_and_the_read_gives_the_recall_up(self):
        """The listener thread can drop the room's recall after the ranks
        agreed that READY is in. The rank must reach the next agreement."""
        mgr = self._scripted()
        cache, _, puller = self._build(host_capacity=32, mgr=mgr)
        host = cache.cache_controller.host
        req = self._req()
        puller.advance([req])
        # READY is in for the agreement, and gone when the read is to start.
        mgr.prefix_ready.side_effect = [(11, "10.0.0.2:5"), None]
        puller.advance([req])
        self.assertNotIn(ROOM, puller._plans)
        self.assertEqual(puller.stats["failed"], 1)
        mgr.start_host_read.assert_not_called()
        puller.advance([req])
        self.assertEqual(host.available_size(), 32)

    def test_dropped_landing_is_freed_once_the_read_settled_on_every_rank(self):
        """Host evictions depend on the free room, so the ranks' host pools
        must stay equal: a rank whose read settled keeps the landing while
        another rank's read can still write its own."""
        mgr = self._scripted()
        cache, _, puller = self._build(host_capacity=32, mgr=mgr)
        host = cache.cache_controller.host
        req = self._req()
        self._advance_to_pulling(puller, mgr, req)
        mgr.read_done = False
        puller.drop_departed([])
        with patch.object(
            puller._lockstep, "reduce", side_effect=lambda rows, op: [[0]] * len(rows)
        ):
            puller.free_settled_host_landings()
        self.assertEqual(host.available_size(), 16)
        self.assertTrue(puller.holds_host_landing_pages())
        puller.free_settled_host_landings()
        self.assertEqual(host.available_size(), 32)
        self.assertFalse(puller.holds_host_landing_pages())

    def test_offer_waits_unmatched_until_every_rank_holds_it(self):
        """The channel hands an offer out once and its listener thread can
        drop the room at any time. A match splits nodes, so no rank matches
        before every rank holds the offer, and none loses it in between."""
        mgr = self._scripted()
        cache, _, puller = self._build(host_capacity=32, mgr=mgr)
        req = self._req()
        with (
            patch.object(
                puller._lockstep, "reduce", side_effect=lambda rows, op: [[0, 0, 0, 0]]
            ),
            patch.object(cache, "match_prefix") as match,
        ):
            puller.advance([req])
        match.assert_not_called()
        self.assertNotIn(ROOM, puller._plans)
        puller.advance([req])
        self.assertIn(ROOM, puller._plans)
        mgr.take_prefix_offer.assert_called_once_with(ROOM)
        # An offer of a request that left the queue is dropped with it.
        puller._offered[ROOM + 1] = [1]
        puller.drop_departed([req])
        self.assertEqual(puller._offered, {})

    def test_dropped_request_frees_its_landing_once_no_read_is_pending(self):
        mgr = self._scripted()
        cache, _, puller = self._build(host_capacity=32, mgr=mgr)
        host = cache.cache_controller.host
        req = self._req()
        self._advance_to_pulling(puller, mgr, req)
        self.assertEqual(host.available_size(), 16)
        # Raiden cannot cancel the read: the pages stay reserved, and a memory
        # release must wait.
        puller.drop_departed([])
        self.assertEqual(host.available_size(), 16)
        with self.assertRaisesRegex(RuntimeError, "prefix reads are in flight"):
            puller.release_all()
        mgr.read_done = False
        puller.release_all()
        self.assertEqual(host.available_size(), 32)
        self.assertFalse(puller.holds_host_landing_pages())


class TestPrefillEndAfterAQueueStep(CustomTestCase):
    def test_bootstrapped_requests_are_committed_and_departed_plans_dropped(self):
        extension = PrefillDramOffloadExtension.__new__(PrefillDramOffloadExtension)
        extension.recall, extension.push = MagicMock(), MagicMock()
        extension.recall.insert_recalled_host_prefix.return_value = True
        extension.after_bootstrap_step(bootstrapped=["a"], queued=["b"])
        extension.recall.insert_recalled_host_prefix.assert_called_once_with("a")
        extension.push.note_decode_of.assert_called_once_with("a", recalled=True)
        extension.recall.drop_departed.assert_called_once_with(["b"])


class TestDecodeQueueHooks(DecodeQueueHooksFixture, CustomTestCase):
    """The plugin's wrappers of the decode queues: a request's offer is made
    just before its PULL_REQ is sent, and the offers of requests that left the
    transfer queue are released in the same step, without any change to the
    core queues."""

    def test_offer_is_made_before_the_pull_req_is_sent(self):
        prealloc = SimpleNamespace(tree_cache=MagicMock())
        decode_req = MagicMock()
        calls = []
        upstream = MagicMock(side_effect=lambda *a, **k: calls.append("send"))
        offload = SimpleNamespace(source=MagicMock())
        offload.source.offer_host_prefix.side_effect = lambda dr: calls.append(
            ("offer", dr)
        )
        with patch.object(core_hooks, "_extension", return_value=offload) as lookup:
            core_hooks._send_kv_metadata(
                upstream, prealloc, decode_req, [1, 2], 8, None, decode_prefix_len=16
            )
        self.assertEqual(calls, [("offer", decode_req), "send"])
        lookup.assert_called_once_with(prealloc.tree_cache)
        upstream.assert_called_once_with(
            prealloc, decode_req, [1, 2], 8, None, decode_prefix_len=16
        )

    def test_request_the_decode_holds_no_page_of_offers_nothing(self):
        # A host receive sends decode_prefix_len=0.
        for prefix_len in (0, 7):
            with self.subTest(prefix_len=prefix_len):
                upstream = MagicMock()
                with patch.object(core_hooks, "_extension") as offer:
                    core_hooks._send_kv_metadata(
                        upstream,
                        MagicMock(),
                        MagicMock(),
                        [1, 2],
                        8,
                        decode_prefix_len=prefix_len,
                    )
                offer.assert_not_called()
                upstream.assert_called_once()

    def test_decode_never_caches_its_own_generation_requests(self):
        req = SimpleNamespace(skip_radix_cache_insert=False)
        upstream = MagicMock(return_value="added")
        self.assertEqual(
            core_hooks._decode_add(upstream, "queue", req, 1, key=2), "added"
        )
        self.assertTrue(req.skip_radix_cache_insert)
        upstream.assert_called_once_with("queue", req, 1, key=2)


class TestEnginesAreValidatedAtStart(CustomTestCase):
    """Both roles check the settings as soon as the transfer manager holds the
    model's KV pools, so an unsupported engine never serves a request."""

    def test_each_role_validates_its_manager_then_extends_its_l2_cache(self):
        scheduler = SimpleNamespace(tree_cache="prefill-tree")
        queues = {
            "prefill": (
                core_hooks._prefill_init_kv_manager,
                SimpleNamespace(scheduler=scheduler),
                "prefill-tree",
            ),
            "decode": (
                core_hooks._decode_init_kv_manager,
                SimpleNamespace(
                    scheduler=scheduler,
                    tree_cache="decode-tree",
                    transfer_queue="transfer-queue",
                ),
                "decode-tree",
            ),
        }
        for role, (hook, queue, tree) in queues.items():
            attach_kwargs = dict(cache=tree, channel="channel", scheduler=scheduler)
            if role == "decode":
                attach_kwargs["decode_queue"] = "transfer-queue"
            calls = []
            manager = SimpleNamespace(dram_offload_channel="channel")
            upstream = MagicMock(return_value=manager)
            with (
                self.subTest(role=role),
                patch.object(
                    core_hooks,
                    "validate_engine",
                    side_effect=lambda **kw: calls.append(("validate", kw)),
                ),
                patch.object(
                    core_hooks,
                    "attach_dram_offload_extension",
                    side_effect=lambda **kw: calls.append(("attach", kw)),
                ),
            ):
                self.assertIs(hook(upstream, queue), manager)
            # Refusals come first: nothing is attached to a refused engine.
            self.assertEqual(
                calls,
                [
                    (
                        "validate",
                        dict(
                            role=role,
                            scheduler=scheduler,
                            kv_manager=manager,
                            tree_cache=tree,
                        ),
                    ),
                    ("attach", attach_kwargs),
                ],
            )

    def test_a_refusal_stops_the_engine(self):
        queue = SimpleNamespace(scheduler=SimpleNamespace(tree_cache=None))
        refusal = startup_validation.UnsupportedDramOffloadConfig("no")
        with (
            patch.object(
                core_hooks,
                "validate_engine",
                side_effect=refusal,
            ),
            patch.object(core_hooks, "attach_dram_offload_extension") as attach,
        ):
            with self.assertRaises(ValueError):
                core_hooks._prefill_init_kv_manager(MagicMock(), queue)
        attach.assert_not_called()

    def test_both_startup_hooks_are_installed(self):
        installed = {
            (owner.__name__, name) for owner, name, _ in core_hooks._patch_targets()
        }
        self.assertIn(("PrefillBootstrapQueue", "_init_kv_manager"), installed)
        self.assertIn(("DecodePreallocQueue", "_init_kv_manager"), installed)
        self.assertIn(("DecodeTransferQueue", "pop_transferred"), installed)


class TestDecodeOffersWhatItHoldsInRam(CustomTestCase):
    """The decode names the deepest pushed prefix of the prompt that is whole
    in its host RAM and pins it, and offers nothing when it holds none. The manager is
    autospecced, so a call the real manager would reject fails here."""

    def _source(self):
        cache, _ = full_cache(kv_size=64, host_capacity=64)
        manager = create_autospec(DramOffloadChannel, instance=True)
        source = DecodeDramPrefixOfferer(
            offload_context(cache=cache, channel=manager), SimpleNamespace()
        )
        return cache, manager, source

    def _pushed(self, cache, tokens):
        """A victim a prefill pushed: a host-only path, as the receiver inserts it."""
        host = cache.cache_controller.host
        radix_host_ops.insert_host_prefix(
            cache, RadixKey(array("q", tokens)), host.alloc(len(tokens)), 1
        )

    def _decode_req(self, tokens):
        return SimpleNamespace(
            req=SimpleNamespace(
                bootstrap_room=ROOM,
                origin_input_ids=tokens,
                extra_key=None,
                cache_salt=None,
            )
        )

    def test_held_prefix_is_offered_page_for_page_and_pinned(self):
        cache, manager, source = self._source()
        victim = list(range(300, 316))
        self._pushed(cache, victim)
        source.offer_host_prefix(self._decode_req(victim + [1, 2, 3, 4]))
        manager.offer_resident_prefix.assert_called_once()
        room, pages = manager.offer_resident_prefix.call_args.args
        self.assertEqual((room, len(pages)), (ROOM, 16))
        self.assertIn(ROOM, source.offers)
        # Pinned for the prefill's read.
        self.assertEqual(cache.evict_host(16), 0)
        # A request is offered once.
        source.offer_host_prefix(self._decode_req(victim + [1, 2, 3, 4]))
        manager.offer_resident_prefix.assert_called_once()

    def test_shared_prefix_of_an_offer_keeps_its_host_pages(self):
        """The prefill reads pages of the whole offered path. Under write-back
        a host eviction reclaims the host copy of an unlocked node that is
        also on the device, so every node of the path stays locked."""
        cache, allocator = full_cache(kv_size=64, host_capacity=32)
        manager = create_autospec(DramOffloadChannel, instance=True)
        source = DecodeDramPrefixOfferer(
            offload_context(cache=cache, channel=manager), SimpleNamespace()
        )
        victim = [1, 2, 3, 4, 10, 11, 12, 13]
        insert_tokens(cache, allocator, victim)
        cache.evict(EvictParams(num_tokens=8))
        insert_tokens(cache, allocator, [1, 2, 3, 4, 20, 21, 22, 23])
        source.offer_host_prefix(self._decode_req(victim + [99]))
        _, pages, _ = (*manager.offer_resident_prefix.call_args.args, None)[:3]
        self.assertEqual(len(pages), 8)
        cache.evict_host(4)
        leaf = cache.tree_core.node_by_id(source.offers[ROOM][0][0])
        self.assertEqual(
            radix_host_ops.host_pages(cache, cache.cache_controller.host, leaf), pages
        )

    def test_prompt_the_decode_does_not_hold_offers_nothing(self):
        cache, manager, source = self._source()
        self._pushed(cache, list(range(300, 316)))
        source.offer_host_prefix(self._decode_req(list(range(900, 920))))
        manager.offer_resident_prefix.assert_not_called()
        self.assertEqual(source.offers, {})
        self.assertEqual(cache.evict_host(16), 16)

    def test_offer_never_covers_the_last_prompt_token(self):
        # The prefill computes at least one token, so a prompt equal to the
        # victim is offered one page short.
        cache, manager, source = self._source()
        victim = list(range(300, 316))
        self._pushed(cache, victim)
        source.offer_host_prefix(self._decode_req(victim))
        _, pages = manager.offer_resident_prefix.call_args.args
        self.assertLess(len(pages), 16)


class TestHostOfferLeavesWithItsRequest(CustomTestCase):
    """The decode pins the host pages it offered. The pin must go once the
    request has left the transfer queue and the prefill's read has settled,
    or every offered victim stays unevictable and the flush never runs."""

    def _queue_with_offer(self):
        decode_req = SimpleNamespace(req=SimpleNamespace(bootstrap_room=ROOM))
        queue = MagicMock(queue=[decode_req], _deferred_releases=[])
        manager = self.manager = MagicMock()
        manager.host_prefix_pending.return_value = False
        ctx = DramOffloadWorkerContext(
            cache=queue.tree_cache,
            host_pool=None,
            page_size=1,
            channel=manager,
            lockstep=RankLockstep(),
        )
        source = DecodeDramPrefixOfferer(ctx, queue)
        source.offers = {ROOM: [("node", "receipt")]}
        return queue, source, decode_req

    def test_offer_is_kept_while_the_request_is_queued_or_deferred(self):
        queue, source, decode_req = self._queue_with_offer()
        source.close_departed_offers()
        self.assertEqual(source._released, set())
        # Held for a deferred release: still owned by the queue.
        queue.queue.clear()
        queue._deferred_releases.append((decode_req, 0.0, 0.0, 0, 0))
        source.close_departed_offers()
        self.assertEqual(source._released, set())
        self.assertTrue(source.offers)

    def test_offer_is_released_once_the_request_left_and_the_read_settled(self):
        queue, source, _ = self._queue_with_offer()
        queue.queue.clear()
        self.manager.host_prefix_pending.return_value = True
        source.close_departed_offers()
        self.assertEqual(source._released, {ROOM})
        # The prefill still reads the pages: the pin stays.
        source.unpin_settled_host_offers()
        queue.tree_cache.dec_host_lock_ref.assert_not_called()
        self.manager.host_prefix_pending.return_value = False
        source.unpin_settled_host_offers()
        queue.tree_cache.dec_host_lock_ref.assert_called_once_with("node", "receipt")
        self.assertFalse(source.offers)

    def test_cache_reset_releases_the_offers_of_a_cleared_queue(self):
        queue, source, _ = self._queue_with_offer()
        # A reset while the request is queued must refuse, not drop a pin.
        with self.assertRaisesRegex(RuntimeError, "host prefix offers"):
            source.before_cache_reset()
        queue.queue.clear()
        source.before_cache_reset()
        self.assertFalse(source.offers)


class TestPushThenRecallBetweenTwoEngines(CustomTestCase):
    """Both ends of the extension against each other: real radix caches, real
    workers and real raiden channels, with the messages handed over by the
    test. Only the native transfer is a double, completed by the test."""

    def _engine(self, mgr):
        cache, allocator = full_cache(kv_size=64, host_capacity=32)
        channel = mgr.dram_offload_channel
        channel.bind_host_pool(cache.cache_controller.host)
        # The engine drain polls the host reads first; nothing else runs here.
        mgr._drain_engine = channel.drain_host_reads
        return SimpleNamespace(
            mgr=mgr,
            cache=cache,
            allocator=allocator,
            host=cache.cache_controller.host,
            ctx=offload_context(cache=cache, channel=channel),
            delivered=0,
        )

    def _deliver(self, src, dst):
        """Hand ``dst`` every message ``src`` sent since the last call."""
        calls = src.mgr._send_multipart_locked.call_args_list
        for call in calls[src.delivered :]:
            dst.mgr._dispatch(call.args[1])
        src.delivered = len(calls)

    def test_cold_branch_is_pushed_to_the_decode_and_recalled_when_it_returns(self):
        p, d = self._engine(prefill_manager()), self._engine(decode_manager())
        prefill = PrefillDramOffloadExtension(p.ctx)
        queue = SimpleNamespace(queue=[], _deferred_releases=[])
        decode = DecodeDramOffloadExtension(d.ctx, queue)
        session = list(range(100, 116))

        # The prefill served the session to this decode and holds it in host RAM.
        insert_tokens(p.cache, p.allocator, session)
        p.mgr._prefill.decode_addr[1] = ("10.0.0.2", 5555)
        prefill.commit_recall(pushed_req(room=1, tokens=session))
        p.cache.evict(EvictParams(num_tokens=16))
        self.assertEqual(host_hit(p.cache, session), 16)

        # Push: the prefill offers the pages, the decode reads and inserts
        # them, and its answer makes the prefill's copy the next victim.
        prefill.before_bootstrap_step([], serving=True)
        self._deliver(p, d)
        for _ in range(decode_dram_victim_receiver._CADENCE):
            decode.advance()
        self.assertTrue(decode.holds_host_pages_outside_tree())
        d.host.transfer.complete("received")
        p.host.transfer.complete("sent")
        for _ in range(decode_dram_victim_receiver._CADENCE):
            decode.advance()
        self.assertEqual(host_hit(d.cache, session), 16)
        self._deliver(d, p)
        prefill.before_bootstrap_step([], serving=True)
        self.assertEqual(prefill.push.stats["stored"], 1)
        self.assertFalse(prefill.holds_host_pages_outside_tree())
        p.cache.evict_host(16, FULL)
        self.assertEqual(host_hit(p.cache, session), 0)

        # The session returns: the decode offers what it holds in its PULL_REQ.
        req = pushed_req(room=ROOM, tokens=session + [1, 2, 3, 4])
        decode_req = SimpleNamespace(req=req)
        queue.queue.append(decode_req)
        decode.source.offer_host_prefix(decode_req)
        sender = sender_on(p.mgr)
        d.mgr.register_decode_request(ROOM, [21, 22, 23], [], None, [PREFILL_INFO])
        with SWITCH.override(True):
            self._deliver(d, p)
        self.assertEqual(sender.poll(), KVPoll.Bootstrapping)

        # Recall: PREFIX_PULL, PREFIX_READY, then the read into prefill host RAM.
        prefill.before_bootstrap_step([req], serving=True)
        self._deliver(p, d)
        self._deliver(d, p)
        prefill.before_bootstrap_step([req], serving=True)
        self.assertTrue(prefill.holds_host_pages_outside_tree())
        p.host.transfer.complete("received")
        d.host.transfer.complete("sent")
        prefill.before_bootstrap_step([req], serving=True)
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)
        prefill.commit_recall(req)
        self.assertEqual(prefill.recall.stats["pulled_tokens"], 16)
        self.assertEqual(host_hit(p.cache, session), 16)

        # The decode's pin goes once the request left its queue.
        self.assertTrue(decode.holds_host_pages_outside_tree())
        queue.queue.clear()
        decode.advance()
        self.assertFalse(decode.holds_host_pages_outside_tree())
        # What was just recalled is not offered back to the decode that gave it.
        sent = p.mgr._send_multipart_locked.call_count
        prefill.before_bootstrap_step([], serving=True)
        self.assertEqual(p.mgr._send_multipart_locked.call_count, sent)
        self.assertFalse(prefill.holds_host_pages_outside_tree())


if __name__ == "__main__":
    unittest.main()
