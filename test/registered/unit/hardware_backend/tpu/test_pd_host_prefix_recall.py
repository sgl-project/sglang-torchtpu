"""Edge cases of the recall from decode RAM into the prefill's RAM
(sglang_tpu.pd_host_prefix_recall.HostPrefixPuller): no host room, room made
by reclaiming copies the device still holds, a READY that does not match the
plan, and a read that fails after it started. Every path must leave the
prefill computing the prompt, with no page leaked and no lock held."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _load_sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_vp = _load_sibling("test_pd_victim_push")

from array import array  # noqa: E402

from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from sglang_tpu import pd_host_prefix_recall  # noqa: E402
from sglang_tpu.disaggregation.raiden.decode_dram_offload import (  # noqa: E402
    DecodeDramOffload,
)
from sglang_tpu.pd_host_cache import insert_host_prefix  # noqa: E402
from sglang_tpu.pd_host_prefix_recall import (  # noqa: E402
    HostPrefixPuller,
    HostPrefixSource,
)

ROOM = 7


def _manager(*, offer_pages: int, refused: bool = False):
    mgr = MagicMock()
    mgr.take_prefix_offer.return_value = list(range(offer_pages))
    mgr.ask_prefix_pull.return_value = True
    mgr.prefix_pull_ready.return_value = (
        f"{ROOM}#px",
        0 if refused else 11,
        "10.0.0.2:5",
    )
    mgr.prefix_read_state.return_value = "unknown"
    return mgr


class _HostRecallEdgeCasesFixture:
    def _build(self, *, host_capacity: int, mgr):
        cache, allocator = _vp.full_cache(kv_size=64, host_capacity=host_capacity)
        cache.tree_core.is_write_back = cache.is_write_back = False
        scheduler = SimpleNamespace(
            tree_cache=cache,
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(1)),
            attn_tp_cpu_group=None,
            attn_cp_cpu_group=None,
        )
        puller = HostPrefixPuller(kv_manager=mgr, scheduler=scheduler)
        puller._max_inflight_tokens = 1 << 20
        return cache, allocator, puller

    def _fill_host_with_a_device_copy(self, cache, allocator, *, lock: bool):
        tokens = list(range(100, 116))
        _vp.insert_tokens(cache, allocator, tokens)
        node = _vp.root_node(cache, 100)
        host = cache.cache_controller.host
        cache.tree_core.commit_backup(node.id, host.alloc(len(node.key)), {})
        cache.tree_core._update_duplicate_tracking(node)
        if lock:
            cache.inc_host_lock_ref(node.id)
        self.assertEqual(host.available_size(), 0)
        return host

    def _req(self):
        return SimpleNamespace(
            bootstrap_room=ROOM,
            origin_input_ids=list(range(200, 232)),
            extra_key=None,
            cache_salt=None,
        )


class _RecallLifecycleFixture:
    def _scripted(self):
        """A manager that offers 16 pages once and lets the test set the READY
        and the read state as the handshake proceeds."""
        mgr = _manager(offer_pages=16)
        mgr.prefix_pull_ready.return_value = None
        mgr.prefix_pull_resolved.return_value = False
        return mgr

    def _advance_to_pulling(self, puller, mgr, req):
        puller.advance([req])
        plan = puller._plans[ROOM]
        self.assertEqual((plan.state, plan.new_indices), ("asked", None))
        mgr.ask_prefix_pull.assert_called_once()
        # Host pages are reserved only once READY is in.
        puller.advance([req])
        self.assertIsNone(plan.new_indices)
        mgr.prefix_pull_ready.return_value = (f"{ROOM}#px", 11, "10.0.0.2:5")
        mgr.prefix_read_state.return_value = "pending"
        puller.advance([req])
        self.assertEqual(plan.state, "pulling")
        self.assertEqual(len(plan.new_indices), 16)
        return plan


class TestRecallLifecycle(
    _RecallLifecycleFixture, _HostRecallEdgeCasesFixture, CustomTestCase
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
        mgr.prefix_read_state.return_value = "done"
        puller.advance([req])
        self.assertTrue(plan.agreed_done)
        mgr.resolve_prefix_pull.assert_called_once_with(ROOM)
        puller.commit(req)
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

    def test_ranks_planning_different_ranges_give_up_together(self):
        mgr = self._scripted()
        cache, _, puller = self._build(host_capacity=32, mgr=mgr)
        req = self._req()
        puller._plan(req)
        self.assertIn(ROOM, puller._plans)
        # Another rank planned 8 pages: MIN and MAX of the plan rows differ.
        with patch.object(
            puller, "_all_ranks_reduce", side_effect=[[[1, 0, 8]], [[1, 0, 16]]]
        ):
            puller._agree_on_plans([ROOM])
        self.assertNotIn(ROOM, puller._plans)
        mgr.resolve_prefix_pull.assert_called_with(ROOM)
        self.assertFalse(puller.has_pending_deferred_releases())

    def test_dropped_request_frees_its_landing_once_no_read_is_pending(self):
        mgr = self._scripted()
        cache, _, puller = self._build(host_capacity=32, mgr=mgr)
        host = cache.cache_controller.host
        req = self._req()
        self._advance_to_pulling(puller, mgr, req)
        self.assertEqual(host.available_size(), 16)
        # Raiden cannot cancel the read: the pages stay reserved, and a memory
        # release must wait.
        puller.release(req)
        self.assertEqual(host.available_size(), 16)
        with self.assertRaisesRegex(RuntimeError, "prefix reads are in flight"):
            puller.release_all()
        mgr.prefix_read_state.return_value = "failed"
        puller.release_all()
        self.assertEqual(host.available_size(), 32)
        self.assertFalse(puller.has_pending_deferred_releases())


class _DecodeQueueHooksFixture:
    def _queue(self, *rids):
        reqs = [
            SimpleNamespace(req=SimpleNamespace(rid=rid, bootstrap_room=n))
            for n, rid in enumerate(rids)
        ]
        return SimpleNamespace(queue=reqs), reqs


class TestDecodeQueueHooks(_DecodeQueueHooksFixture, CustomTestCase):
    """The plugin's wrappers of the decode queues: a request's offer is made
    just before its PULL_REQ is sent, and the offers of requests that left the
    transfer queue are released in the same step, without any change to the
    core queues."""

    def test_offer_is_made_before_the_pull_req_is_sent(self):
        prealloc = SimpleNamespace(transfer_queue=MagicMock())
        decode_req = MagicMock()
        calls = []
        upstream = MagicMock(side_effect=lambda *a, **k: calls.append("send"))
        with patch.object(
            _vp.victim_push,
            "offer_host_prefix",
            side_effect=lambda q, dr: calls.append(("offer", q, dr)),
        ):
            _vp.victim_push._send_kv_metadata(
                upstream, prealloc, decode_req, [1, 2], 8, None, decode_prefix_len=16
            )
        self.assertEqual(
            calls, [("offer", prealloc.transfer_queue, decode_req), "send"]
        )
        upstream.assert_called_once_with(
            prealloc, decode_req, [1, 2], 8, None, decode_prefix_len=16
        )

    def test_request_the_decode_holds_no_page_of_offers_nothing(self):
        # A host receive sends decode_prefix_len=0.
        for prefix_len in (0, 7):
            with self.subTest(prefix_len=prefix_len):
                upstream = MagicMock()
                with patch.object(_vp.victim_push, "offer_host_prefix") as offer:
                    _vp.victim_push._send_kv_metadata(
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
            _vp.victim_push._decode_add(upstream, "queue", req, 1, key=2), "added"
        )
        self.assertTrue(req.skip_radix_cache_insert)
        upstream.assert_called_once_with("queue", req, 1, key=2)


class TestEnginesAreValidatedAtStart(CustomTestCase):
    """Both roles check the settings as soon as the transfer manager holds the
    model's KV pools, so an unsupported engine never serves a request."""

    def test_each_role_validates_the_manager_it_just_built(self):
        scheduler = SimpleNamespace(tree_cache="prefill-tree")
        prefill_queue = SimpleNamespace(scheduler=scheduler)
        decode_queue = SimpleNamespace(scheduler=scheduler, tree_cache="decode-tree")
        upstream = MagicMock(return_value="manager")
        with patch.object(_vp.victim_push.pd_spill_validate, "validate_engine") as v:
            self.assertEqual(
                _vp.victim_push._prefill_init_kv_manager(upstream, prefill_queue),
                "manager",
            )
            v.assert_called_once_with(
                role="prefill",
                scheduler=scheduler,
                kv_manager="manager",
                tree_cache="prefill-tree",
            )
            v.reset_mock()
            self.assertEqual(
                _vp.victim_push._decode_init_kv_manager(upstream, decode_queue),
                "manager",
            )
            v.assert_called_once_with(
                role="decode",
                scheduler=scheduler,
                kv_manager="manager",
                tree_cache="decode-tree",
            )

    def test_a_refusal_stops_the_engine(self):
        queue = SimpleNamespace(scheduler=SimpleNamespace(tree_cache=None))
        refusal = _vp.victim_push.pd_spill_validate.UnsupportedSpillConfig("no")
        with patch.object(
            _vp.victim_push.pd_spill_validate, "validate_engine", side_effect=refusal
        ):
            with self.assertRaises(ValueError):
                _vp.victim_push._prefill_init_kv_manager(MagicMock(), queue)

    def test_both_startup_hooks_are_installed(self):
        installed = {
            (owner.__name__, name)
            for owner, name, _ in _vp.victim_push._patch_targets()
        }
        self.assertIn(("PrefillBootstrapQueue", "_init_kv_manager"), installed)
        self.assertIn(("DecodePreallocQueue", "_init_kv_manager"), installed)
        self.assertIn(("DecodeTransferQueue", "pop_transferred"), installed)


class TestDecodeOffersWhatItHoldsInRam(CustomTestCase):
    """The decode names the deepest pushed prefix of the prompt that is whole
    in its host RAM and pins it, and offers nothing when it holds none. The manager is
    autospecced, so a call the real manager would reject fails here."""

    def _source(self):
        cache, allocator = _vp.full_cache(kv_size=64, host_capacity=64)
        manager = create_autospec(DecodeDramOffload, instance=True)
        queue = SimpleNamespace(
            tree_cache=cache,
            scheduler=SimpleNamespace(
                token_to_kv_pool_allocator=allocator,
                attn_tp_cpu_group=None,
                attn_cp_cpu_group=None,
            ),
        )
        source = HostPrefixSource(queue, manager)
        manager.bind_prefix_host_pool.assert_called_once_with(
            cache.cache_controller.host
        )
        return cache, manager, source

    def _pushed(self, cache, tokens):
        """A victim a prefill pushed: a host-only path, as the receiver inserts it."""
        host = cache.cache_controller.host
        insert_host_prefix(
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
        source.offer(self._decode_req(victim + [1, 2, 3, 4]))
        manager.offer_resident_prefix.assert_called_once()
        room, pages = manager.offer_resident_prefix.call_args.args
        self.assertEqual((room, len(pages)), (ROOM, 16))
        self.assertIn(ROOM, source.offers)
        # Pinned for the prefill's read.
        self.assertEqual(cache.evict_host(16), 0)
        # A request is offered once.
        source.offer(self._decode_req(victim + [1, 2, 3, 4]))
        manager.offer_resident_prefix.assert_called_once()

    def test_prompt_the_decode_does_not_hold_offers_nothing(self):
        cache, manager, source = self._source()
        self._pushed(cache, list(range(300, 316)))
        source.offer(self._decode_req(list(range(900, 920))))
        manager.offer_resident_prefix.assert_not_called()
        self.assertEqual(source.offers, {})
        self.assertEqual(cache.evict_host(16), 16)

    def test_offer_never_covers_the_last_prompt_token(self):
        # The prefill computes at least one token, so a prompt equal to the
        # victim is offered one page short.
        cache, manager, source = self._source()
        victim = list(range(300, 316))
        self._pushed(cache, victim)
        source.offer(self._decode_req(victim))
        _, pages = manager.offer_resident_prefix.call_args.args
        self.assertLess(len(pages), 16)


class TestHostOfferLeavesWithItsRequest(CustomTestCase):
    """The decode pins the host pages it offered. The pin must go once the
    request has left the transfer queue and the prefill's read has settled,
    or every offered victim stays unevictable and the flush never runs."""

    def _queue_with_offer(self):
        decode_req = SimpleNamespace(req=SimpleNamespace(bootstrap_room=ROOM))
        queue = MagicMock(queue=[decode_req], _deferred_releases=[])
        manager = MagicMock()
        manager.host_prefix_pending.return_value = False
        source = HostPrefixSource.__new__(HostPrefixSource)
        source.cache, source.manager = queue.tree_cache, manager
        source.offers = {ROOM: ("node", "receipt")}
        source.released = set()
        source.groups = []
        return queue, source, decode_req

    def _sources(self, queue, source):
        return patch.object(pd_host_prefix_recall, "_sources", {queue: source})

    def test_offer_is_kept_while_the_request_is_queued_or_deferred(self):
        queue, source, decode_req = self._queue_with_offer()
        with self._sources(queue, source):
            pd_host_prefix_recall.release_departed_host_offers(queue)
            self.assertEqual(source.released, set())
            # Held for a deferred release: still owned by the queue.
            queue.queue.clear()
            queue._deferred_releases.append((decode_req, 0.0, 0.0, 0, 0))
            pd_host_prefix_recall.release_departed_host_offers(queue)
            self.assertEqual(source.released, set())
            self.assertTrue(pd_host_prefix_recall.host_offers_pending(queue))

    def test_offer_is_released_once_the_request_left_and_the_read_settled(self):
        queue, source, _ = self._queue_with_offer()
        with self._sources(queue, source):
            queue.queue.clear()
            source.manager.host_prefix_pending.return_value = True
            pd_host_prefix_recall.release_departed_host_offers(queue)
            self.assertEqual(source.released, {ROOM})
            # The prefill still reads the pages: the pin stays.
            pd_host_prefix_recall.progress_host_offers(queue)
            queue.tree_cache.dec_host_lock_ref.assert_not_called()
            source.manager.host_prefix_pending.return_value = False
            pd_host_prefix_recall.progress_host_offers(queue)
            queue.tree_cache.dec_host_lock_ref.assert_called_once_with(
                "node", "receipt"
            )
            self.assertFalse(pd_host_prefix_recall.host_offers_pending(queue))

    def test_cache_reset_releases_the_offers_of_a_cleared_queue(self):
        queue, source, _ = self._queue_with_offer()
        with self._sources(queue, source):
            # A reset while the request is queued must refuse, not drop a pin.
            with self.assertRaisesRegex(RuntimeError, "host prefix offers"):
                pd_host_prefix_recall.reset_host_offers(queue.tree_cache)
            queue.queue.clear()
            pd_host_prefix_recall.reset_host_offers(queue.tree_cache)
            self.assertFalse(pd_host_prefix_recall.host_offers_pending(queue))


if __name__ == "__main__":
    unittest.main()
