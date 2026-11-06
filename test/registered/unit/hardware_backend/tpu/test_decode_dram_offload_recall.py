"""Edge cases of the recall from decode RAM into the prefill's RAM
(sglang_tpu.prefill_dram_prefix_puller.PrefillDramPrefixPuller): no host room, room made
by reclaiming copies the device still holds, a READY that does not match the
plan, and a read that fails after it started. Every path must leave the
prefill computing the prompt, with no page leaked and no lock held."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec, patch

import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.decode_dram_offload import (
    decode_dram_victim_receiver,
    dram_offload_hooks,
    radix_host_ops,
    startup_validation,
)
from sglang_tpu.disaggregation.decode_dram_offload.decode_dram_prefix_offerer import (
    DecodeDramPrefixOfferer,
)
from sglang_tpu.disaggregation.decode_dram_offload.dram_offload_hooks import (
    DecodeDramOffloadExtension,
    PrefillDramOffloadExtension,
)
from sglang_tpu.disaggregation.decode_dram_offload.prefill_dram_prefix_puller import (
    PrefillDramPrefixPuller,
)
from sglang_tpu.disaggregation.decode_dram_offload.prefill_dram_victim_pusher import (
    PrefillDramVictimPusher,
)
from sglang_tpu.disaggregation.decode_dram_offload.rank_lockstep import (
    DramOffloadWorkerContext,
    RankLockstep,
)
from sglang_tpu.disaggregation.raiden.dram_offload_channel import DramOffloadChannel
from sglang_tpu.testing.decode_dram_offload import (
    FULL,
    HYBRID_PAGE,
    PREFILL_INFO,
    ROOM,
    SWITCH,
    DecodeQueueHooksFixture,
    HostRecallEdgeCasesFixture,
    RecallLifecycleFixture,
    decode_manager,
    device_hit,
    full_cache,
    host_hit,
    hybrid_cache,
    insert_tokens,
    offload_context,
    prefill_manager,
    pushed_req,
    recall_channel,
    root_node,
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
        puller.advance([req], waiting_reqs=[])
        self.assertIsNone(plan.agreed_done)
        mgr.resolve_prefix_pull.assert_not_called()
        mgr.read_done = True
        puller.advance([req], waiting_reqs=[])
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
        puller.advance([], waiting_reqs=[])
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
            puller.advance([req], waiting_reqs=[])
        mgr.ask_prefix_pull.assert_called_once_with(ROOM, 4, 12)
        plan = puller._plans[ROOM]
        self.assertEqual((plan.first_page, plan.num_pages), (4, 12))
        self.assertEqual(plan.remote, list(range(4, 16)))
        mgr.ready, mgr.read_done = (11, "10.0.0.2:5", 0, ""), None
        puller.advance([req], waiting_reqs=[])
        mgr.read_done = True
        puller.advance([req], waiting_reqs=[])
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
                    puller.advance([self._req()], waiting_reqs=[])
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
        puller.advance([req], waiting_reqs=[])
        # READY is in for the agreement, and gone when the read is to start.
        mgr.prefix_ready.side_effect = [(11, "10.0.0.2:5"), None]
        puller.advance([req], waiting_reqs=[])
        self.assertNotIn(ROOM, puller._plans)
        self.assertEqual(puller.stats["failed"], 1)
        mgr.start_host_read.assert_not_called()
        puller.advance([req], waiting_reqs=[])
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
            puller.advance([req], waiting_reqs=[])
        match.assert_not_called()
        self.assertNotIn(ROOM, puller._plans)
        puller.advance([req], waiting_reqs=[])
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
        # release is refused until the read settled.
        puller.drop_departed([])
        self.assertEqual(host.available_size(), 16)
        with self.assertRaisesRegex(RuntimeError, "not idle"):
            puller.release_all()
        mgr.read_done = False
        puller.free_settled_host_landings()
        puller.release_all()
        self.assertEqual(host.available_size(), 32)
        self.assertFalse(puller.holds_host_landing_pages())


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
        with patch.object(
            dram_offload_hooks, "_extension", return_value=offload
        ) as lookup:
            dram_offload_hooks._send_kv_metadata(
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
                with patch.object(dram_offload_hooks, "_extension") as offer:
                    dram_offload_hooks._send_kv_metadata(
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
            dram_offload_hooks._decode_add(upstream, "queue", req, 1, key=2), "added"
        )
        self.assertTrue(req.skip_radix_cache_insert)
        upstream.assert_called_once_with("queue", req, 1, key=2)


class TestEnginesAreValidatedAtStart(CustomTestCase):
    """Both roles check the settings before the transfer manager is built, so
    an unsupported engine never serves a request."""

    def test_each_role_validates_then_extends_its_l2_cache(self):
        scheduler = SimpleNamespace(tree_cache="prefill-tree")
        queues = {
            "prefill": (
                dram_offload_hooks._prefill_init_kv_manager,
                SimpleNamespace(scheduler=scheduler),
                "prefill-tree",
            ),
            "decode": (
                dram_offload_hooks._decode_init_kv_manager,
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
            upstream = MagicMock(
                side_effect=lambda q: calls.append("manager") or manager
            )
            with (
                self.subTest(role=role),
                patch.object(
                    dram_offload_hooks,
                    "validate_engine_args",
                    side_effect=lambda **kw: calls.append(("validate", kw)),
                ),
                patch.object(
                    dram_offload_hooks,
                    "attach_dram_offload_extension",
                    side_effect=lambda **kw: calls.append(("attach", kw)),
                ),
            ):
                self.assertIs(hook(upstream, queue), manager)
            # Refusals come first: nothing is built for or attached to a
            # refused engine.
            self.assertEqual(
                calls,
                [
                    ("validate", dict(role=role, scheduler=scheduler)),
                    "manager",
                    ("attach", attach_kwargs),
                ],
            )

    def test_a_refusal_stops_the_engine_before_the_manager_is_built(self):
        """A pipeline transfer started by the manager held the engine's log
        pipe for minutes after a refusal."""
        queue = SimpleNamespace(scheduler=SimpleNamespace(tree_cache=None))
        refusal = startup_validation.UnsupportedDramOffloadConfig("no")
        upstream = MagicMock()
        for hook in (
            dram_offload_hooks._prefill_init_kv_manager,
            dram_offload_hooks._decode_init_kv_manager,
        ):
            with (
                self.subTest(hook=hook.__name__),
                patch.object(
                    dram_offload_hooks, "validate_engine_args", side_effect=refusal
                ),
                patch.object(
                    dram_offload_hooks, "attach_dram_offload_extension"
                ) as attach,
            ):
                with self.assertRaises(ValueError):
                    hook(upstream, queue)
            upstream.assert_not_called()
            attach.assert_not_called()

    def test_both_startup_hooks_are_installed(self):
        installed = {
            (owner.__name__, name)
            for owner, name, _ in dram_offload_hooks._patch_targets()
        }
        self.assertIn(("PrefillBootstrapQueue", "_init_kv_manager"), installed)
        self.assertIn(("DecodePreallocQueue", "_init_kv_manager"), installed)
        self.assertIn(("DecodeTransferQueue", "pop_transferred"), installed)
        self.assertIn(
            ("SchedulerWeightUpdaterManager", "release_memory_occupation"), installed
        )


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
        room, pages, swa_pages = manager.offer_resident_prefix.call_args.args
        self.assertEqual(swa_pages, [])
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
        _, pages, _ = manager.offer_resident_prefix.call_args.args
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

    def test_cache_reset_refuses_while_an_offer_is_pinned(self):
        queue, source, _ = self._queue_with_offer()
        # A reset while the request is queued must refuse, not drop a pin.
        with self.assertRaisesRegex(RuntimeError, "not idle"):
            source.release_all()
        queue.queue.clear()
        source.settle_offers()
        source.release_all()
        self.assertFalse(source.offers)


class TestPushThenRecallBetweenTwoEngines(CustomTestCase):
    """Both ends of the extension against each other: real radix caches, real
    workers and real raiden channels, with the messages handed over by the
    test. Only the native transfer is a double, completed by the test."""

    def _engine(self, mgr, cache_and_allocator=None, *, page_size=1):
        cache, allocator = cache_and_allocator or full_cache(
            kv_size=64, host_capacity=32
        )
        channel = mgr.dram_offload_channel
        ctx = offload_context(cache=cache, channel=channel, page_size=page_size)
        channel.bind_host_pools(ctx.host_pool, ctx.swa_host_pool)
        return SimpleNamespace(
            mgr=mgr,
            cache=cache,
            allocator=allocator,
            host=ctx.host_pool,
            swa_host=ctx.swa_host_pool,
            ctx=ctx,
            delivered=0,
        )

    def _deliver(self, src, dst):
        """Hand ``dst`` every message ``src`` sent since the last call."""
        calls = src.mgr._send_multipart_locked.call_args_list
        for call in calls[src.delivered :]:
            dst.mgr._dispatch(call.args[1])
        src.delivered = len(calls)

    @staticmethod
    def _prefill_step(prefill, reqs):
        """What the bootstrap-queue hook runs before core's step."""
        prefill.recall.advance(reqs, waiting_reqs=[])
        prefill.push.advance(serving=True)

    @staticmethod
    def _commit(prefill, req):
        """What the bootstrap-queue hook runs for a request that left bootstrap."""
        prefill.push.note_decode_of(
            req, recalled=prefill.recall.insert_recalled_host_prefix(req)
        )

    @staticmethod
    def _decode_step(decode):
        """What the decode-queue hook runs every decode iteration."""
        decode.source.settle_offers()
        decode.landing.advance()

    def test_cold_branch_is_pushed_to_the_decode_and_recalled_when_it_returns(self):
        p, d = self._engine(prefill_manager()), self._engine(decode_manager())
        prefill = PrefillDramOffloadExtension(
            recall=PrefillDramPrefixPuller(p.ctx), push=PrefillDramVictimPusher(p.ctx)
        )
        queue = SimpleNamespace(queue=[], _deferred_releases=[])
        decode = DecodeDramOffloadExtension(
            landing=decode_dram_victim_receiver.DecodeDramVictimReceiver(d.ctx),
            source=DecodeDramPrefixOfferer(d.ctx, queue),
        )
        holds = dram_offload_hooks._holds_host_pages_outside_tree
        session = list(range(100, 116))

        # The prefill served the session to this decode and holds it in host RAM.
        insert_tokens(p.cache, p.allocator, session)
        p.mgr._prefill.decode_addr[1] = ("10.0.0.2", 5555)
        self._commit(prefill, pushed_req(room=1, tokens=session))
        p.cache.evict(EvictParams(num_tokens=16))
        self.assertEqual(host_hit(p.cache, session), 16)

        # Push: the prefill offers the pages, the decode reads and inserts
        # them, and its answer makes the prefill's copy the next victim.
        self._prefill_step(prefill, [])
        self._deliver(p, d)
        for _ in range(decode_dram_victim_receiver._CADENCE):
            self._decode_step(decode)
        self.assertTrue(holds(decode))
        d.host.transfer.complete("received")
        p.host.transfer.complete("sent")
        for _ in range(decode_dram_victim_receiver._CADENCE):
            self._decode_step(decode)
        self.assertEqual(host_hit(d.cache, session), 16)
        self._deliver(d, p)
        self._prefill_step(prefill, [])
        self.assertEqual(prefill.push.stats["stored"], 1)
        self.assertFalse(holds(prefill))
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
        self._prefill_step(prefill, [req])
        self._deliver(p, d)
        self._deliver(d, p)
        self._prefill_step(prefill, [req])
        self.assertTrue(holds(prefill))
        p.host.transfer.complete("received")
        d.host.transfer.complete("sent")
        self._prefill_step(prefill, [req])
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)
        self._commit(prefill, req)
        self.assertEqual(prefill.recall.stats["pulled_tokens"], 16)
        self.assertEqual(host_hit(p.cache, session), 16)

        # The decode's pin goes once the request left its queue.
        self.assertTrue(holds(decode))
        queue.queue.clear()
        self._decode_step(decode)
        self.assertFalse(holds(decode))
        # What was just recalled is not offered back to the decode that gave it.
        sent = p.mgr._send_multipart_locked.call_count
        self._prefill_step(prefill, [])
        self.assertEqual(p.mgr._send_multipart_locked.call_count, sent)
        self.assertFalse(holds(prefill))


class TestHybridWindowTravelsWithTheBranch(TestPushThenRecallBetweenTwoEngines):
    """A hybrid sliding-window model moves the window that ends where the
    branch ends, through the SWA host pools, with the full pages."""

    def test_hybrid_window_over_a_held_window_returns_its_pages(self):
        """A branch that returns one page longer brings a window that overlaps
        the one its path already holds. The overlap goes back to the SWA host
        pool, not to the L3 release queue, which nothing drains here."""
        page, window = HYBRID_PAGE, 2 * HYBRID_PAGE
        cache, _ = hybrid_cache(window=window)
        full = cache.cache_controller.host
        swa = radix_host_ops.swa_host_pool(cache)
        session = list(range(6 * page))
        for pages in range(1, 7):
            tokens = session[: pages * page]
            radix_host_ops.insert_host_prefix(
                cache,
                RadixKey(array("q", tokens)),
                full.alloc(len(tokens)),
                page,
                swa_indices=swa.alloc(min(len(tokens), window)),
            )
        self.assertEqual(full.used, len(session))
        self.assertLessEqual(swa.used, len(session))
        cache.evict_host(len(session), FULL)
        cache.evict_host(len(session), ComponentType.SWA)
        self.assertEqual((full.used, swa.used), (0, 0))

    def test_hybrid_branch_is_pushed_and_recalled_with_its_window(self):
        page, window = HYBRID_PAGE, 2 * HYBRID_PAGE
        p = self._engine(prefill_manager(), hybrid_cache(window=window), page_size=page)
        d = self._engine(decode_manager(), hybrid_cache(window=window), page_size=page)
        prefill = PrefillDramOffloadExtension(
            recall=PrefillDramPrefixPuller(p.ctx), push=PrefillDramVictimPusher(p.ctx)
        )
        queue = SimpleNamespace(queue=[], _deferred_releases=[])
        decode = DecodeDramOffloadExtension(
            landing=decode_dram_victim_receiver.DecodeDramVictimReceiver(d.ctx),
            source=DecodeDramPrefixOfferer(d.ctx, queue),
        )
        session = list(range(4 * page))

        # The prefill holds the session and its last window in host RAM only.
        radix_host_ops.insert_host_prefix(
            p.cache,
            RadixKey(array("q", session)),
            p.host.alloc(len(session)),
            page,
            swa_indices=p.swa_host.alloc(window),
        )
        p.mgr._prefill.decode_addr[1] = ("10.0.0.2", 5555)
        self._commit(prefill, pushed_req(room=1, tokens=session))

        # Push: four full pages and the two window pages, each to its pool.
        self._prefill_step(prefill, [])
        self._deliver(p, d)
        (push,) = d.mgr.dram_offload_channel.received_pushes().values()
        self.assertEqual((len(push.blocks), len(push.swa_blocks)), (4, 2))
        for _ in range(decode_dram_victim_receiver._CADENCE):
            self._decode_step(decode)
        self.assertEqual((d.host.used, d.swa_host.used), (len(session), window))
        for pool, status in (
            (d.host, "received"),
            (d.swa_host, "received"),
            (p.host, "sent"),
            (p.swa_host, "sent"),
        ):
            pool.transfer.complete(status)
        for _ in range(decode_dram_victim_receiver._CADENCE):
            self._decode_step(decode)
        self._deliver(d, p)
        self._prefill_step(prefill, [])
        self.assertEqual(prefill.push.stats["stored"], 1)
        p.cache.evict_host(len(session), FULL)
        self.assertEqual(p.host.used, 0)

        # The session returns one page longer: the decode offers its four
        # pages with the window that ends there, and the prefill takes both.
        req = pushed_req(room=ROOM, tokens=session + list(range(1000, 1000 + page)))
        decode_req = SimpleNamespace(req=req)
        queue.queue.append(decode_req)
        decode.source.offer_host_prefix(decode_req)
        sender = sender_on(p.mgr)
        d.mgr.register_decode_request(ROOM, [21, 22, 23], [], None, [PREFILL_INFO])
        with SWITCH.override(True):
            self._deliver(d, p)
        self._prefill_step(prefill, [req])
        self._deliver(p, d)
        self._deliver(d, p)
        self._prefill_step(prefill, [req])
        plan = prefill.recall._plans[ROOM]
        self.assertEqual(plan.read_ids, [f"{ROOM}#px", f"{ROOM}#pxs"])
        for pool, status in (
            (p.host, "received"),
            (p.swa_host, "received"),
            (d.host, "sent"),
            (d.swa_host, "sent"),
        ):
            pool.transfer.complete(status)
        self._prefill_step(prefill, [req])
        self.assertEqual(sender.poll(), KVPoll.WaitingForInput)
        self._commit(prefill, req)
        self.assertEqual(prefill.recall.stats["pulled_tokens"], len(session))
        self.assertEqual((p.host.used, p.swa_host.used), (len(session), window))


class TestReceivedPagesBackUpDeviceOnlyNodes(CustomTestCase):
    """Under write-back, a recalled branch inserted below a node that is on
    the device without a host copy would leave a subtree that eviction can
    neither back up (no host room) nor drop (the branch's host pages are
    locked by the request); the received pages of that node become its
    backup instead of being freed."""

    def test_device_only_prefix_is_backed_by_the_received_pages(self):
        cache, allocator = full_cache(kv_size=64, host_capacity=64)
        host = cache.cache_controller.host
        session = list(range(100, 108))
        insert_tokens(cache, allocator, session[:4])
        head = root_node(cache, session[0])
        self.assertFalse(head.backuped)
        stats = {"adopted_backup_tokens": 0}
        radix_host_ops.insert_host_prefix(
            cache, RadixKey(array("q", session)), host.alloc(8), 1, stats=stats
        )
        self.assertTrue(head.backuped)
        self.assertIsNotNone(head.component_data[ComponentType.FULL].value)
        self.assertEqual(host.used, 8)
        self.assertEqual(stats["adopted_backup_tokens"], 4)
        self.assertIn(head.id, cache.tree_core.full_host_duplicates)
        self.assertIsNone(head.write_through_pending_id)
        self.assertEqual((device_hit(cache, session), host_hit(cache, session)), (4, 4))
        # Device eviction demotes the head instead of having to drop its subtree.
        cache.evict(EvictParams(num_tokens=4))
        self.assertIsNone(head.component_data[ComponentType.FULL].value)
        self.assertTrue(head.backuped)
        self.assertEqual((device_hit(cache, session), host_hit(cache, session)), (0, 8))

    def test_backed_prefix_still_frees_its_received_duplicates(self):
        cache, allocator = full_cache(kv_size=64, host_capacity=64)
        host = cache.cache_controller.host
        session = list(range(100, 108))
        radix_host_ops.insert_host_prefix(
            cache, RadixKey(array("q", session[:4])), host.alloc(4), 1
        )
        stats = {"adopted_backup_tokens": 0}
        radix_host_ops.insert_host_prefix(
            cache, RadixKey(array("q", session)), host.alloc(8), 1, stats=stats
        )
        self.assertEqual((host.used, stats["adopted_backup_tokens"]), (8, 0))

    def test_reused_prefix_is_neither_adopted_nor_freed(self):
        cache, allocator = full_cache(kv_size=64, host_capacity=64)
        host = cache.cache_controller.host
        session = list(range(100, 108))
        insert_tokens(cache, allocator, session[:2])
        radix_host_ops.insert_host_prefix(
            cache, RadixKey(array("q", session[:4])), host.alloc(4), 1
        )
        head = root_node(cache, session[0])
        self.assertTrue(head.backuped)
        # The first four pages are the caller's own host-pinned prefix; they
        # are neither freed nor adopted, whatever the path holds.
        used = host.used
        stats = {"adopted_backup_tokens": 0}
        radix_host_ops.insert_host_prefix(
            cache,
            RadixKey(array("q", session)),
            torch.cat([host.alloc(4), host.alloc(4)]),
            1,
            reused_tokens=4,
            stats=stats,
        )
        self.assertEqual(host.used, used + 8)
        self.assertEqual(stats["adopted_backup_tokens"], 0)


# ---- Lower-priority cases: less common paths, wire round trips, wording ----


class TestHostRecallEdgeCasesExtra(HostRecallEdgeCasesFixture, CustomTestCase):
    def _asked_and_ready(self, puller, req):
        """Two steps: the range is agreed and asked for, then READY is acted on."""
        puller.advance([req], waiting_reqs=[])
        self.assertIn(ROOM, puller._plans)
        puller.advance([req], waiting_reqs=[])

    def test_no_host_room_abandons_the_recall_and_computes(self):
        mgr = recall_channel(offer_pages=16)
        cache, allocator, puller = self._build(host_capacity=16, mgr=mgr)
        host = self._fill_host_with_a_device_copy(cache, allocator, lock=True)
        self._asked_and_ready(puller, self._req())
        self.assertEqual(puller.stats["no_memory"], 1)
        self.assertNotIn(ROOM, puller._plans)
        mgr.resolve_prefix_pull.assert_called_with(ROOM)
        mgr.start_host_read.assert_not_called()
        self.assertEqual(host.available_size(), 0)  # nothing leaked, nothing taken
        self.assertFalse(puller.holds_host_landing_pages())

    def test_room_is_made_by_reclaiming_a_copy_the_device_holds(self):
        mgr = recall_channel(offer_pages=16)
        mgr.read_done = None
        cache, allocator, puller = self._build(host_capacity=16, mgr=mgr)
        self._fill_host_with_a_device_copy(cache, allocator, lock=False)
        self._asked_and_ready(puller, self._req())
        plan = puller._plans[ROOM]
        self.assertEqual(puller.stats["no_memory"], 0)
        self.assertEqual(device_hit(cache, list(range(100, 116))), 16)
        self.assertEqual(plan.read_ids, [f"{ROOM}#px"])
        args = mgr.start_host_read.call_args.args
        self.assertEqual(args[0], f"{ROOM}#px")
        self.assertEqual(len(args[4]), 16)  # landing pages named to raiden
        self.assertTrue(puller.holds_host_landing_pages())

    def test_refused_pull_is_abandoned(self):
        mgr = recall_channel(offer_pages=16, refused=True)
        cache, allocator, puller = self._build(host_capacity=32, mgr=mgr)
        host = cache.cache_controller.host
        self._asked_and_ready(puller, self._req())
        self.assertEqual(puller.stats["failed"], 1)
        self.assertNotIn(ROOM, puller._plans)
        mgr.start_host_read.assert_not_called()
        puller.free_settled_host_landings()
        self.assertEqual(host.available_size(), 32)  # landing freed

    def test_read_another_rank_could_not_start_is_given_up_by_all(self):
        mgr = recall_channel(offer_pages=16)
        mgr.read_done = None
        cache, allocator, puller = self._build(host_capacity=32, mgr=mgr)
        host = cache.cache_controller.host
        req = self._req()
        puller.advance([req], waiting_reqs=[])
        reduce = puller._lockstep.reduce
        # The start is the last agreement of the step: another rank reports 0.
        calls = []

        def another_rank_fails_to_start(rows, op):
            calls.append(rows)
            return [[0]] if len(calls) == 4 else reduce(rows, op)

        with patch.object(
            puller._lockstep, "reduce", side_effect=another_rank_fails_to_start
        ):
            puller.advance([req], waiting_reqs=[])
        self.assertNotIn(ROOM, puller._plans)
        mgr.resolve_prefix_pull.assert_called_with(ROOM)
        # This rank's read did start: its landing stays owned until it settles.
        self.assertEqual(host.available_size(), 16)
        self.assertTrue(puller.holds_host_landing_pages())
        mgr.read_done = False
        puller.free_settled_host_landings()
        self.assertEqual(host.available_size(), 32)

    def test_read_that_fails_after_start_frees_the_landing_once_settled(self):
        mgr = recall_channel(offer_pages=16)
        mgr.read_done = None
        cache, allocator, puller = self._build(host_capacity=32, mgr=mgr)
        host = cache.cache_controller.host
        req = self._req()
        self._asked_and_ready(puller, req)
        free_before = host.available_size()
        # The decode leaves early (prealloc done) while raiden may still write.
        puller.insert_recalled_host_prefix(req)
        self.assertNotIn(ROOM, puller._plans)
        self.assertEqual(host.available_size(), free_before)  # still owned
        self.assertTrue(puller.holds_host_landing_pages())
        mgr.read_done = False
        puller.free_settled_host_landings()
        self.assertEqual(host.available_size(), free_before + 16)
        self.assertFalse(puller.holds_host_landing_pages())

    def test_inflight_recalls_never_take_more_than_their_share_of_host_ram(self):
        mgr = recall_channel(offer_pages=16)
        cache, _, puller = self._build(host_capacity=32, mgr=mgr)
        puller._max_inflight_tokens = 16
        first, second = self._req(), self._req()
        second.bootstrap_room = ROOM + 1
        puller.advance([first, second], waiting_reqs=[])
        self.assertIn(ROOM, puller._plans)
        self.assertNotIn(ROOM + 1, puller._plans)
        self.assertEqual(puller.stats["over_budget"], 1)
        mgr.resolve_prefix_pull.assert_called_once_with(ROOM + 1)
        mgr.ask_prefix_pull.assert_called_once_with(ROOM, 0, 16)


class TestDecodeQueueHooksExtra(DecodeQueueHooksFixture, CustomTestCase):
    def test_pop_releases_the_offers_of_departed_requests(self):
        for rids in (("a", "b"), ()):
            with self.subTest(queued=rids):
                queue, _ = self._queue(*rids)
                calls = []
                upstream = MagicMock(
                    side_effect=lambda q, r: calls.append("pop") or ["transferred"]
                )
                offload = SimpleNamespace(source=MagicMock())
                offload.source.close_departed_offers.side_effect = lambda: calls.append(
                    "release"
                )
                with patch.object(
                    dram_offload_hooks, "_extension", return_value=offload
                ):
                    result = dram_offload_hooks._pop_transferred(upstream, queue, ["b"])
                self.assertEqual(result, ["transferred"])
                # A request can leave by abort while the queue is empty of work.
                self.assertEqual(calls, ["pop", "release"])
                upstream.assert_called_once_with(queue, ["b"])


if __name__ == "__main__":
    unittest.main()
