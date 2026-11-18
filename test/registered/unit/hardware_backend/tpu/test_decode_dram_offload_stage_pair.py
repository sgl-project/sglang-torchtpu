"""A pipeline-parallel prefill against a tensor-parallel decode, both ends of
the offload wired as the engines wire them (dram_offload_hooks.py): real radix
caches on every prefill stage, real workers, real raiden channels, and the
hooks the pipeline loop calls. The test moves the messages, runs the ring the
way scheduler_pp_mixin does, and completes the native transfers, which are
doubles."""

import unittest
from array import array
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.decode_dram_offload import (
    decode_stage_branches,
    dram_offload_hooks,
    radix_host_ops,
    stage_pool_builder,
)
from sglang_tpu.disaggregation.decode_dram_offload.stage_layout_store import (
    StagePoolLayout,
)
from sglang_tpu.disaggregation.raiden.common import _tcp_endpoint
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.testing.decode_dram_offload import (
    FULL,
    HYBRID_PAGE,
    FakeHostTransfer,
    decode_manager,
    full_cache,
    host_hit,
    hybrid_cache,
    insert_tokens,
    prefill_manager,
    pushed_req,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

STAGES = 2
LAG = 3
DECODE_PORT = 5555


def _layout(host_pool, *, stage, stages, window=False):
    return StagePoolLayout(stage, stages, (("bfloat16", (1, 2, 64)),), 1024, window)


class _Stage:
    """One stage of the prefill: its cache, manager, bootstrap queue and the
    scheduler surface the hooks read."""

    def __init__(self, index, cache_and_allocator=None, page_size=1):
        self.cache, self.allocator = cache_and_allocator or full_cache(
            kv_size=64, host_capacity=32
        )
        self.mgr = prefill_manager()
        self.mgr.pp_rank, self.mgr.pp_size = index, STAGES
        self.mgr.rank_port = 6000 + index
        self.channel = self.mgr.dram_offload_channel
        self.queue = SimpleNamespace(queue=[])
        self.scheduler = SimpleNamespace(
            tree_cache=self.cache,
            # The engine attaches the extension while it builds the queue.
            disagg_prefill_bootstrap_queue=None,
            waiting_queue=[],
            running_batch=SimpleNamespace(is_empty=lambda: True),
            token_to_kv_pool_allocator=SimpleNamespace(page_size=page_size),
        )
        self.queue.scheduler = self.scheduler
        self.extension = dram_offload_hooks.attach_dram_offload_extension(
            cache=self.cache, channel=self.channel, scheduler=self.scheduler
        )
        self.scheduler.disagg_prefill_bootstrap_queue = self.queue
        self.host = self.extension.push._ctx.host_pool
        self.swa_host = self.extension.push._ctx.swa_host_pool
        self.draft = self.extension.push._draft_pool
        self.delivered = 0

    @property
    def endpoint(self):
        return _tcp_endpoint(ip=self.mgr.local_ip, port=self.mgr.rank_port)


class _Pair:
    def __init__(self, test, make_cache=None, page_size=1, draft=False):
        if draft:
            # Every stage runs draft layers: a sidecar pool that follows the
            # full-attention pool's page ids.
            patcher = patch.object(
                dram_offload_hooks,
                "draft_sidecar_host_pools",
                lambda cache: (SimpleNamespace(transfer=FakeHostTransfer()), None),
            )
            patcher.start()
            test.addCleanup(patcher.stop)
            allow = tpu_envs.SGLANG_PD_OFFLOAD_ALLOW_DRAFT.override(True)
            allow.__enter__()
            test.addCleanup(allow.__exit__, None, None, None)
        for target, value in (
            ("stage_pool_layout", _layout),
            ("build_stage_pool", lambda layout, pages, device: FakeHostTransfer()),
        ):
            patcher = patch.object(stage_pool_builder, target, value)
            patcher.start()
            test.addCleanup(patcher.stop)
        override = tpu_envs.SGLANG_RAIDEN_PIPELINE_TRANSFER.override(True)
        override.__enter__()
        test.addCleanup(override.__exit__, None, None, None)

        self.stages = [
            _Stage(index, make_cache() if make_cache else None, page_size)
            for index in range(STAGES)
        ]
        self.decode = SimpleNamespace(mgr=decode_manager(), delivered=0)
        mgr = self.decode.mgr
        mgr.attn_tp_size, mgr.rank_port = 1, DECODE_PORT
        self.decode.endpoint = _tcp_endpoint(ip=mgr.local_ip, port=DECODE_PORT)
        self.decode.queue = SimpleNamespace(queue=[], _deferred_releases=[])
        own_pool = SimpleNamespace(
            transfer=SimpleNamespace(
                num_host_pages=64, _device_tensors=[[SimpleNamespace(device="cpu")]]
            )
        )
        self.decode.extension = dram_offload_hooks._stage_decode_extension(
            mgr.dram_offload_channel,
            SimpleNamespace(attn_tp_cpu_group=None),
            own_pool,
            page_size,
            self.decode.queue,
        )
        self.popped = []
        self._ring = deque([None] * LAG)

    # ----- transport -------------------------------------------------------------

    def deliver(self):
        """Hand every message sent since the last call to its endpoint."""
        ends = {stage.endpoint: stage for stage in self.stages}
        ends[self.decode.endpoint] = self.decode
        moved = True
        while moved:
            moved = False
            for end in list(ends.values()):
                calls = end.mgr._send_multipart_locked.call_args_list
                for call in calls[end.delivered :]:
                    ends[call.args[0]].mgr._dispatch(call.args[1])
                    moved = True
                end.delivered = len(calls)

    def decode_asks_for(self, room):
        """The decode's pipeline request for a room reaches every stage."""
        channel = self.decode.mgr.dram_offload_channel
        consumer = dict(
            attn_tp_size=1, attn_tp_rank=0, **channel.pipeline_consumer_fields(room)
        )
        for stage in self.stages:
            stage.channel.on_pipeline_request(room, self.decode.endpoint, consumer)

    # ----- the loops -------------------------------------------------------------

    def ring_step(self, n=1, ready=()):
        """Micro-batch slots of the prefill pipeline: every stage folds its
        ready and failed request ids, and the lists of ``LAG`` slots ago are
        applied by every stage. ``ready``: request ids whose transfer state
        would let them leave bootstrap."""
        for _ in range(n):
            good = bad = None
            for stage in self.stages:
                mine = dram_offload_hooks._get_rids(
                    lambda scheduler, queue: (
                        [
                            r.rid
                            for r in queue
                            if r.rid in ready and self._open(stage, r)
                        ],
                        [],
                    ),
                    stage.scheduler,
                    stage.queue.queue,
                )
                good = mine[0] if good is None else list(set(good) & set(mine[0]))
                bad = mine[1] if bad is None else list(set(bad) | set(mine[1]))
            self._ring.append([good, bad])
            consensus = self._ring.popleft()
            for stage in self.stages:
                if consensus is not None:
                    consensus = dram_offload_hooks._process_bootstrapped_queue(
                        lambda scheduler, rids, stage=stage: self._process(stage, rids),
                        stage.scheduler,
                        consensus,
                    )
            self.deliver()

    @staticmethod
    def _open(stage, req):
        with stage.channel._lock:
            return not stage.channel.prefix_pull_open_locked(req.bootstrap_room)

    def _process(self, stage, rids):
        """scheduler.process_bootstrapped_queue: pop by consensus, pass on
        the popped request ids."""

        def pop(queue, *, pp_good_rids, pp_bad_rids, return_failed_reqs=True):
            good = [r for r in queue.queue if r.rid in pp_good_rids]
            queue.queue[:] = [r for r in queue.queue if r.rid not in pp_good_rids]
            return good, []

        good, _ = dram_offload_hooks._pop_bootstrapped(
            pop, stage.queue, pp_good_rids=rids[0], pp_bad_rids=rids[1]
        )
        if good:
            self.popped.append((stage.mgr.pp_rank, [r.rid for r in good]))
        return [[r.rid for r in good], []]

    def decode_step(self, rounds=1):
        for _ in range(rounds * decode_stage_branches._CADENCE):
            self.decode.extension.source.settle_offers()
            self.decode.extension.landing.advance()
        self.deliver()

    # ----- shortcuts ---------------------------------------------------------------

    def request(self, room, tokens):
        reqs = []
        for stage in self.stages:
            req = pushed_req(room=room, tokens=tokens)
            req.rid = f"rid-{room}"
            stage.queue.queue.append(req)
            # As the request's sender does: the room is live on this stage.
            stage.mgr.request_status[room] = "bootstrapping"
            reqs.append(req)
        return reqs

    def stage_transfers(self, status):
        for stage in self.stages:
            stage.host.transfer.complete(status)
            if stage.swa_host is not None:
                stage.swa_host.transfer.complete(status)
            if stage.draft is not None:
                stage.draft.transfer.complete(status)

    def decode_transfers(self, status):
        for transfer in self.decode.extension.landing._pools._pools.values():
            transfer.complete(status)

    def holds(self):
        check = dram_offload_hooks._holds_host_pages_outside_tree
        return [check(s.extension) for s in self.stages], check(self.decode.extension)


class TestPipelinePrefillAgainstTensorParallelDecode(CustomTestCase):
    SESSION = list(range(100, 116))

    def _serve_and_push(self, pair):
        """The prefill serves the session to the decode, holds it in host
        RAM, and pushes it under host pressure."""
        for stage in pair.stages:
            insert_tokens(stage.cache, stage.allocator, self.SESSION)
        pair.request(1, self.SESSION)
        pair.decode_asks_for(1)
        pair.ring_step(LAG + 2, ready={"rid-1"})
        self.assertEqual(sorted(pair.popped), [(0, ["rid-1"]), (1, ["rid-1"])])
        for stage in pair.stages:
            stage.cache.evict(EvictParams(num_tokens=16))
            self.assertEqual(host_hit(stage.cache, self.SESSION), 16)
            stage.scheduler.waiting_queue.append("serving")
        # Every stage proposes the branch, then all begin the push.
        pair.ring_step(LAG + 2)
        self.assertEqual(
            [s.extension.workers.push.stats["started"] for s in pair.stages], [1, 1]
        )
        pair.decode_step()

    def test_a_cold_branch_is_pushed_stage_by_stage_and_recalled(self):
        pair = _Pair(self)
        self._serve_and_push(pair)
        landing = pair.decode.extension.landing
        # One pool per stage, each reading that stage's pages into the same ids.
        self.assertEqual(
            sorted(landing._pools._pools), [(0, False, False), (1, False, False)]
        )
        for index, stage in enumerate(pair.stages):
            (read,) = landing._pools.transfer(index).reads.values()
            (registered,) = stage.host.transfer.registrations.values()
            self.assertEqual(read["remote_block_ids"], registered[1])
            self.assertEqual(read["local_block_ids"], list(range(16)))
        self.assertEqual(pair.holds(), ([True, True], True))

        pair.decode_transfers("received")
        pair.stage_transfers("sent")
        pair.decode_step()
        self.assertEqual(landing.store.used_pages, 16)
        pair.ring_step(LAG + 2)
        for stage in pair.stages:
            self.assertEqual(stage.extension.workers.push.stats["stored"], 1)
            stage.cache.evict_host(16, FULL)
            self.assertEqual(host_hit(stage.cache, self.SESSION), 0)
        self.assertEqual(pair.holds(), ([False, False], False))

        # The session returns. The decode matches it in the stage-layout
        # index and names the pages in its request to every stage.
        tokens = self.SESSION + [1, 2, 3, 4]
        reqs = pair.request(7, tokens)
        decode_req = SimpleNamespace(req=reqs[0])
        pair.decode.queue.queue.append(decode_req)
        pair.decode.extension.source.offer_stage_prefix(
            room=7, tokens=tokens[:-1], scope=(None, None)
        )
        pair.decode_asks_for(7)
        # The room stays in bootstrap while the recall is open.
        pair.ring_step(2 * LAG + 4, ready={"rid-7"})
        self.assertNotIn((0, ["rid-7"]), pair.popped)
        for index, stage in enumerate(pair.stages):
            (read,) = stage.host.transfer.reads.values()
            self.assertEqual(read["remote_block_ids"], list(range(16)))
            registered = landing._pools.transfer(index).registrations
            self.assertEqual([r[1] for r in registered.values()], [list(range(16))])
        pair.stage_transfers("received")
        pair.decode_transfers("sent")
        pair.ring_step(2 * LAG + 4, ready={"rid-7"})
        for stage in pair.stages:
            self.assertEqual(host_hit(stage.cache, self.SESSION), 16)
            self.assertEqual(stage.extension.workers.recall.stats["pulled_pages"], 16)
        self.assertIn((0, ["rid-7"]), pair.popped)
        self.assertIn((1, ["rid-7"]), pair.popped)

        # The decode's pin goes once the request left its queues.
        self.assertTrue(pair.holds()[1])
        pair.decode.queue.queue.clear()
        pair.decode_step()
        self.assertEqual(pair.holds(), ([False, False], False))
        # What was just recalled is not pushed back at that length.
        pair.ring_step(LAG + 3)
        self.assertEqual(
            [s.extension.workers.push.stats["started"] for s in pair.stages], [1, 1]
        )

    def test_a_recall_that_fails_on_one_stage_is_given_up_by_all(self):
        pair = _Pair(self)
        self._serve_and_push(pair)
        pair.decode_transfers("received")
        pair.stage_transfers("sent")
        pair.decode_step()
        pair.ring_step(LAG + 2)
        for stage in pair.stages:
            stage.cache.evict_host(16, FULL)
        free = [stage.host.available_size() for stage in pair.stages]
        tokens = self.SESSION + [1, 2, 3, 4]
        pair.request(7, tokens)
        pair.decode.extension.source.offer_stage_prefix(
            room=7, tokens=tokens[:-1], scope=(None, None)
        )
        pair.decode_asks_for(7)
        pair.ring_step(2 * LAG + 4, ready={"rid-7"})
        pair.stages[0].host.transfer.complete("received")
        pair.stages[1].host.transfer.complete("failed")
        pair.decode_transfers("sent")
        pair.ring_step(2 * LAG + 4, ready={"rid-7"})
        for stage, was_free in zip(pair.stages, free):
            self.assertEqual(host_hit(stage.cache, self.SESSION), 0)
            self.assertEqual(stage.host.available_size(), was_free)
            self.assertEqual(stage.extension.workers.recall.stats["failed"], 1)
        # The prefill computes the prompt: the request leaves bootstrap.
        self.assertIn((0, ["rid-7"]), pair.popped)

    def _stored_then_returning(self, pair, *, evict=True):
        """The branch is stored on the decode; its session returns as room 7."""
        self._serve_and_push(pair)
        pair.decode_transfers("received")
        pair.stage_transfers("sent")
        pair.decode_step()
        pair.ring_step(LAG + 2)
        if evict:
            for stage in pair.stages:
                stage.cache.evict_host(16, FULL)
        tokens = self.SESSION + [1, 2, 3, 4]
        pair.request(7, tokens)
        pair.decode.extension.source.offer_stage_prefix(
            room=7, tokens=tokens[:-1], scope=(None, None)
        )
        pair.decode_asks_for(7)

    def test_a_request_that_leaves_during_its_recall_frees_the_landing(self):
        pair = _Pair(self)
        self._stored_then_returning(pair)
        free = [stage.host.available_size() for stage in pair.stages]
        pair.ring_step(2 * LAG + 4, ready={"rid-7"})
        self.assertTrue(all(s.extension.recall._pulls for s in pair.stages))
        # The request is dropped on every stage in the same slot (an abort).
        for stage in pair.stages:
            stage.queue.queue.clear()
        pair.stage_transfers("received")
        pair.decode_transfers("sent")
        pair.ring_step(2 * LAG + 4)
        for stage, was_free in zip(pair.stages, free):
            self.assertEqual(host_hit(stage.cache, self.SESSION), 0)
            self.assertEqual(stage.host.available_size(), was_free)
            self.assertEqual(stage.extension.recall._pins, {})
        self.assertEqual(pair.holds()[0], [False, False])

    def test_no_stage_pulls_what_every_stage_still_holds(self):
        pair = _Pair(self)
        self._stored_then_returning(pair, evict=False)
        # Offer, then who lacks it, then the request's own bootstrap vote.
        pair.ring_step(3 * LAG + 4, ready={"rid-7"})
        for stage in pair.stages:
            self.assertEqual(stage.host.transfer.reads, {})
            self.assertEqual(stage.extension.workers.recall.stats["held"], 1)
        self.assertIn((0, ["rid-7"]), pair.popped)

    def test_the_decodes_answer_decides_a_push_whose_registration_went_unread(self):
        """The decode reads only the pages it does not hold yet. A stage's
        registration it had no use for ends at the native deadline, reported
        as failed, on a branch the decode stored (seen live on a hybrid model:
        three stored branches were counted as declined)."""
        pair = _Pair(self)
        self._serve_and_push(pair)
        pair.decode_transfers("received")
        pair.stage_transfers("failed")
        pair.decode_step()
        pair.ring_step(LAG + 2)
        for stage in pair.stages:
            self.assertEqual(stage.extension.workers.push.stats["stored"], 1)
        self.assertEqual(pair.decode.extension.landing.store.used_pages, 16)
        self.assertEqual(pair.holds(), ([False, False], False))

    def test_a_push_the_decode_cannot_store_is_declined_on_every_stage(self):
        pair = _Pair(self)
        self._serve_and_push(pair)
        pair.decode_transfers("failed")
        pair.stage_transfers("failed")
        pair.decode_step()
        pair.ring_step(LAG + 2)
        for stage in pair.stages:
            self.assertEqual(stage.extension.workers.push.stats["declined"], 1)
        self.assertEqual(pair.decode.extension.landing.store.used_pages, 0)
        self.assertEqual(pair.holds(), ([False, False], False))


class TestHybridWindowTravelsStageByStage(CustomTestCase):
    """A hybrid sliding-window model: every stage pushes and recalls the
    window that ends where the branch ends, through its own SWA host pool and
    the decode's window pool of that stage."""

    def test_a_hybrid_branch_is_pushed_and_recalled_with_its_window(self):
        page, window = HYBRID_PAGE, 2 * HYBRID_PAGE
        pair = _Pair(self, lambda: hybrid_cache(window=window), page_size=page)
        session = list(range(4 * page))
        for stage in pair.stages:
            radix_host_ops.insert_host_prefix(
                stage.cache,
                RadixKey(array("q", session)),
                stage.host.alloc(len(session)),
                page,
                swa_indices=stage.swa_host.alloc(window),
            )
            stage.scheduler.waiting_queue.append("serving")
        pair.request(1, session)
        pair.decode_asks_for(1)
        pair.ring_step(2 * LAG + 4, ready={"rid-1"})
        pair.decode_step()
        landing = pair.decode.extension.landing
        self.assertEqual(
            sorted(landing._pools._pools),
            [(0, False, False), (0, True, False), (1, False, False), (1, True, False)],
        )
        for index in range(STAGES):
            (full,) = landing._pools.transfer(index).reads.values()
            (swa,) = landing._pools.transfer(index, swa=True).reads.values()
            self.assertEqual(
                (len(full["remote_block_ids"]), len(swa["remote_block_ids"])), (4, 2)
            )
            self.assertEqual(
                (full["local_block_ids"], swa["local_block_ids"]),
                ([0, 1, 2, 3], [0, 1]),
            )
        pair.decode_transfers("received")
        pair.stage_transfers("sent")
        pair.decode_step()
        pair.ring_step(LAG + 2)
        self.assertEqual((landing.store.used_pages, landing.store.used_windows), (4, 1))
        for stage in pair.stages:
            self.assertEqual(stage.extension.workers.push.stats["stored"], 1)
            stage.cache.evict_host(len(session), FULL)
            self.assertEqual(stage.host.used, 0)

        # The session returns one page longer.
        tokens = session + list(range(1000, 1000 + page))
        pair.request(7, tokens)
        pair.decode.extension.source.offer_stage_prefix(
            room=7, tokens=tokens[: 4 * page], scope=(None, None)
        )
        pair.decode_asks_for(7)
        pair.ring_step(2 * LAG + 4, ready={"rid-7"})
        for index, stage in enumerate(pair.stages):
            self.assertEqual(
                sorted(stage.extension.recall._pulls[7].read_ids), ["7#px", "7#pxs"]
            )
            registered = landing._pools.transfer(index, swa=True).registrations
            self.assertEqual([r[1] for r in registered.values()], [[0, 1]])
        pair.stage_transfers("received")
        pair.decode_transfers("sent")
        pair.ring_step(2 * LAG + 4, ready={"rid-7"})
        for stage in pair.stages:
            self.assertEqual(
                (stage.host.used, stage.swa_host.used), (len(session), window)
            )
            self.assertEqual(stage.extension.workers.recall.stats["pulls"], 1)
        self.assertIn((1, ["rid-7"]), pair.popped)


class TestDraftSidecarTravelsWithTheBranch(CustomTestCase):
    """With a draft (opt-in), every stage moves its draft sidecar pool at the
    page ids of the full-attention pages it belongs to."""

    SESSION = list(range(100, 116))

    def _stored(self, pair):
        for stage in pair.stages:
            insert_tokens(stage.cache, stage.allocator, self.SESSION)
        pair.request(1, self.SESSION)
        pair.decode_asks_for(1)
        pair.ring_step(LAG + 2, ready={"rid-1"})
        for stage in pair.stages:
            stage.cache.evict(EvictParams(num_tokens=16))
            stage.scheduler.waiting_queue.append("serving")
        pair.ring_step(LAG + 2)
        pair.decode_step()

    def test_push_and_recall_move_the_sidecar_at_the_same_page_ids(self):
        pair = _Pair(self, draft=True)
        self._stored(pair)
        landing = pair.decode.extension.landing
        self.assertEqual(
            sorted(landing._pools._pools),
            [(0, False, False), (0, False, True), (1, False, False), (1, False, True)],
        )
        for index, stage in enumerate(pair.stages):
            (full,) = stage.host.transfer.registrations.values()
            ((name, draft),) = stage.draft.transfer.registrations.items()
            self.assertTrue(name.endswith(":draft"))
            # Same UUID, same page ids as the full pages.
            self.assertEqual(draft, full)
            (read,) = landing._pools.transfer(index, draft=True).reads.values()
            (source,) = landing._pools.transfer(index).reads.values()
            self.assertEqual(read["remote_block_ids"], full[1])
            self.assertEqual(read["local_block_ids"], source["local_block_ids"])
        pair.decode_transfers("received")
        pair.stage_transfers("sent")
        pair.decode_step()
        pair.ring_step(LAG + 2)
        for stage in pair.stages:
            self.assertEqual(stage.extension.workers.push.stats["stored"], 1)
            stage.cache.evict_host(16, FULL)

        tokens = self.SESSION + [1, 2, 3, 4]
        pair.request(7, tokens)
        pair.decode.extension.source.offer_stage_prefix(
            room=7, tokens=tokens[:-1], scope=(None, None)
        )
        pair.decode_asks_for(7)
        pair.ring_step(2 * LAG + 4, ready={"rid-7"})
        for index, stage in enumerate(pair.stages):
            (full,) = stage.host.transfer.reads.values()
            ((name, draft),) = stage.draft.transfer.reads.items()
            self.assertEqual(name, "7#pxd")
            # The sidecar lands at the host page ids of the landing itself.
            self.assertEqual(draft["local_block_ids"], full["local_block_ids"])
            self.assertEqual(draft["remote_block_ids"], full["remote_block_ids"])
            self.assertEqual(draft["uuid"], full["uuid"])
            registered = landing._pools.transfer(index, draft=True).registrations
            self.assertEqual([r[1] for r in registered.values()], [list(range(16))])
        pair.stage_transfers("received")
        pair.decode_transfers("sent")
        pair.ring_step(2 * LAG + 4, ready={"rid-7"})
        for stage in pair.stages:
            self.assertEqual(host_hit(stage.cache, self.SESSION), 16)
            self.assertEqual(stage.extension.workers.recall.stats["pulls"], 1)

    def test_a_sidecar_read_that_fails_gives_the_recall_up_on_all_stages(self):
        pair = _Pair(self, draft=True)
        self._stored(pair)
        pair.decode_transfers("received")
        pair.stage_transfers("sent")
        pair.decode_step()
        pair.ring_step(LAG + 2)
        for stage in pair.stages:
            stage.cache.evict_host(16, FULL)
        tokens = self.SESSION + [1, 2, 3, 4]
        pair.request(7, tokens)
        pair.decode.extension.source.offer_stage_prefix(
            room=7, tokens=tokens[:-1], scope=(None, None)
        )
        pair.decode_asks_for(7)
        pair.ring_step(2 * LAG + 4, ready={"rid-7"})
        for stage in pair.stages:
            stage.host.transfer.complete("received")
        pair.stages[0].draft.transfer.complete("received")
        pair.stages[1].draft.transfer.complete("failed")
        pair.decode_transfers("sent")
        pair.ring_step(2 * LAG + 4, ready={"rid-7"})
        for stage in pair.stages:
            self.assertEqual(host_hit(stage.cache, self.SESSION), 0)
            self.assertEqual(stage.extension.workers.recall.stats["failed"], 1)

    def test_without_the_opt_in_no_sidecar_is_looked_up(self):
        with patch.object(
            dram_offload_hooks,
            "draft_sidecar_host_pools",
            side_effect=AssertionError("looked up without the opt-in"),
        ):
            pair = _Pair(self)
        self.assertIsNone(pair.stages[0].draft)


class TestPageByteCheck(CustomTestCase):
    """SGLANG_PD_OFFLOAD_VERIFY_PAGES: a recalled page must hold bytes the
    stage pushed for the same tokens."""

    @staticmethod
    def _pool(blocks, layers=2):
        """A host pool whose manager hands out block bytes."""
        manager = SimpleNamespace(
            read_block_bytes=lambda layer, page: blocks[page] + bytes([layer])
        )
        return SimpleNamespace(transfer=SimpleNamespace(_mgr=manager, layer_num=layers))

    def test_pushed_bytes_verify_and_other_bytes_do_not(self):
        from sglang_tpu.disaggregation.decode_dram_offload.stage_prefill_ops import (
            PageByteCheck,
        )

        check = PageByteCheck(page_size=2)
        tokens = [1, 2, 3, 4, 5, 6]
        source = self._pool({10: b"a", 11: b"b", 12: b"c", 20: b"w"})
        check.record("s", tokens, source, [10, 11, 12], "full")
        check.record("s", tokens, source, [20], "window")
        # The same bytes land in other pages of the pool after a recall.
        landed = self._pool({0: b"a", 1: b"b", 2: b"c", 7: b"w"})
        self.assertEqual(check.verify("s", tokens, landed, [0, 1, 2], "full"), 0)
        self.assertEqual(check.verify("s", tokens, landed, [7], "window"), 0)
        # A shorter recall of the same branch: its first two pages.
        self.assertEqual(check.verify("s", tokens[:4], landed, [0, 1], "full"), 0)
        self.assertEqual(check.stats["verified_pages"], 6)
        # Pages in the wrong order, and a window read as full pages.
        self.assertEqual(check.verify("s", tokens, landed, [1, 0, 2], "full"), 2)
        self.assertEqual(check.verify("s", tokens, landed, [7], "full"), 1)
        self.assertEqual(check.stats["mismatched_pages"], 3)
        # Tokens this stage never pushed are unknown, not wrong.
        self.assertEqual(check.verify("s", [9, 9], landed, [0], "full"), 0)
        self.assertEqual(check.verify("other", tokens, landed, [0, 1, 2], "full"), 0)
        self.assertEqual(check.stats["unknown_pages"], 4)

    def test_the_same_prefix_pushed_twice_keeps_both_contents(self):
        """The decode keeps the first bytes it stored for a page; a later push
        of the same tokens computed in other chunks has other bytes."""
        from sglang_tpu.disaggregation.decode_dram_offload.stage_prefill_ops import (
            PageByteCheck,
        )

        check = PageByteCheck(page_size=2)
        check.record("s", [1, 2], self._pool({0: b"first"}), [0], "full")
        check.record("s", [1, 2], self._pool({0: b"second"}), [0], "full")
        for content in (b"first", b"second"):
            self.assertEqual(
                check.verify("s", [1, 2], self._pool({5: content}), [5], "full"), 0
            )
        self.assertEqual(
            check.verify("s", [1, 2], self._pool({5: b"third"}), [5], "full"), 1
        )

    def test_a_pool_without_block_access_is_skipped(self):
        from sglang_tpu.disaggregation.decode_dram_offload.stage_prefill_ops import (
            PageByteCheck,
        )

        check = PageByteCheck(page_size=2)
        pool = SimpleNamespace(transfer=FakeHostTransfer())
        check.record("s", [1, 2], pool, [0], "full")
        self.assertEqual(check.verify("s", [1, 2], pool, [0], "full"), 0)
        self.assertEqual(sum(check.stats.values()), 0)


if __name__ == "__main__":
    unittest.main()
